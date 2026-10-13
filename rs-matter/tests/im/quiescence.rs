/*
 *
 *    Copyright (c) 2026 Project CHIP Authors
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

//! End-to-end tests for the Interaction Model quiescence gate.
//!
//! The tests drive real reads, writes, invokes and subscriptions over the
//! in-memory transport and assert - through the public
//! [`InteractionQuiescence`](rs_matter::im::InteractionQuiescence) handle -
//! that the model is busy while an interaction or an outbound report is parked
//! at an await point, and idle once it completes - all while the server is
//! still being polled, so the assertions prove normal completion rather than
//! the release that happens when the server is dropped. Cancellation is
//! covered by a dedicated test.

use core::cell::{Cell, RefCell};
use core::future::Future;

use embassy_futures::block_on;
use embassy_futures::join::join;
use embassy_futures::select::select;

use rs_matter::crypto::Crypto;
use rs_matter::dm::clusters::net_comm::{DummyNetworks, NetworkType};
use rs_matter::dm::networks::wireless::NoopWirelessNetCtl;
use rs_matter::dm::{
    AsyncHandler, AttrDetails, CmdDetails, EmptyHandler, HandlerContext, InvokeContext,
    InvokeReply, LifecycleOp, MatchContext, Metadata, Node, ReadContext, ReadReply, ReportContext,
    ReportDataHandler, WriteContext,
};
use rs_matter::error::Error;
use rs_matter::im::client::{ImClient, SubscribeOutcome, TxOutcome};
use rs_matter::im::encoding::{AttrResp, ReportDataResp};
use rs_matter::im::{
    AttrDataTag, AttrPath, CmdDataTag, GenericPath, IMStatusCode, InteractionModel,
    InteractionModelState,
};
use rs_matter::persist::DummyKvBlobStore;
use rs_matter::respond::Responder;
use rs_matter::tlv::{TLVElement, TLVTag, TLVWrite, ToTLV};
use rs_matter::transport::exchange::MatterBuffers;
use rs_matter::utils::select::Coalesce;
use rs_matter::utils::sync::Notification;

use crate::common::e2e::im::echo_cluster;
use crate::common::e2e::{new_default_runner, E2eRunner, E2E_EVENTS_BUF_SIZE};
use crate::common::init_env_logger;

/// Run a test body on a thread with a large stack: the future state-machines
/// of a responder plus a client (and, for the report test, a second
/// Interaction Model) do not fit the default test-thread stack.
fn with_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

/// A single-use async gate.
///
/// The first `enter()` (while armed) notifies `entered` and parks until
/// `release()` is called. This is the "handler is blocked mid-operation" point
/// the tests assert on.
struct Gate {
    armed: Cell<bool>,
    entered: Notification,
    release: Notification,
}

impl Gate {
    fn new(armed: bool) -> Self {
        Self {
            armed: Cell::new(armed),
            entered: Notification::new(),
            release: Notification::new(),
        }
    }

    async fn enter(&self) {
        if self.armed.replace(false) {
            self.entered.notify();
            self.release.wait().await;
        }
    }

    fn release(&self) {
        self.release.notify();
    }

    async fn wait_entered(&self) {
        self.entered.wait().await;
    }
}

/// Wraps a real data model and parks selected read/write/invoke operations in
/// [`Gate`], so a test can observe the model while the operation is in flight.
///
/// `run` also waits on a trigger notification and then performs an "idle
/// commit": mark every cluster dirty and bump `ConfigurationVersion` with no
/// await in between.
struct GatedHandler<'a, H> {
    inner: H,
    gate: &'a Gate,
    trigger: &'a Notification,
}

impl<'a, H> GatedHandler<'a, H> {
    fn new(inner: H, gate: &'a Gate, trigger: &'a Notification) -> Self {
        Self {
            inner,
            gate,
            trigger,
        }
    }

    fn gates_read(attr: &AttrDetails) -> bool {
        attr.endpoint_id == 1
            && attr.cluster_id == echo_cluster::ID
            && attr.attr_id == echo_cluster::AttributesDiscriminants::Att1 as u32
    }

    fn gates_write(attr: &AttrDetails) -> bool {
        attr.endpoint_id == 1
            && attr.cluster_id == echo_cluster::ID
            && attr.attr_id == echo_cluster::AttributesDiscriminants::AttWrite as u32
    }

    fn gates_invoke(cmd: &CmdDetails) -> bool {
        cmd.endpoint_id == 1
            && cmd.cluster_id == echo_cluster::ID
            && cmd.cmd_id == echo_cluster::Commands::EchoReq as u32
    }
}

impl<H: AsyncHandler> AsyncHandler for GatedHandler<'_, H> {
    fn read(
        &self,
        ctx: impl ReadContext,
        reply: impl ReadReply,
    ) -> impl Future<Output = Result<(), Error>> {
        let gated = Self::gates_read(ctx.attr());

        async move {
            if gated {
                self.gate.enter().await;
            }

            self.inner.read(ctx, reply).await
        }
    }

    fn write(&self, ctx: impl WriteContext) -> impl Future<Output = Result<(), Error>> {
        let gated = Self::gates_write(ctx.attr());

        async move {
            if gated {
                self.gate.enter().await;
            }

            self.inner.write(ctx).await
        }
    }

    fn invoke(
        &self,
        ctx: impl InvokeContext,
        reply: impl InvokeReply,
    ) -> impl Future<Output = Result<(), Error>> {
        let gated = Self::gates_invoke(ctx.cmd());

        async move {
            if gated {
                self.gate.enter().await;
            }

            self.inner.invoke(ctx, reply).await
        }
    }

    fn bump_dataver(&self, ctx: impl MatchContext) {
        self.inner.bump_dataver(ctx)
    }

    fn lifecycle(&self, ctx: impl HandlerContext, op: LifecycleOp) -> Result<(), Error> {
        self.inner.lifecycle(ctx, op)
    }

    async fn run(&self, ctx: impl HandlerContext) -> Result<(), Error> {
        loop {
            self.trigger.wait().await;

            // One idle commit: every cluster is dirty and the configuration
            // version changes, with no await between the two - so a single
            // report action has to carry both.
            ctx.notify_all_changed();
            ctx.matter().bump_configuration_version(ctx.kv(), &ctx)?;
        }
    }
}

impl<H: Metadata> Metadata for GatedHandler<'_, H> {
    fn access<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Node<'_>) -> R,
    {
        self.inner.access(f)
    }
}

/// A controller-side [`ReportDataHandler`] that records the attribute paths of
/// every chunk, blocks the first chunk until released (proving the server is
/// parked in its inter-chunk acknowledgement wait), and signals `done` on the
/// last chunk.
struct GatedReports<'a> {
    gate: &'a Gate,
    chunks: Cell<u32>,
    attrs: RefCell<Vec<(u16, u32, u32)>>,
    done: Notification,
}

impl<'a> GatedReports<'a> {
    fn new(gate: &'a Gate) -> Self {
        Self {
            gate,
            chunks: Cell::new(0),
            attrs: RefCell::new(Vec::new()),
            done: Notification::new(),
        }
    }
}

impl ReportDataHandler for GatedReports<'_> {
    async fn handle_report(
        &self,
        _ctx: impl ReportContext,
        report: &ReportDataResp<'_>,
    ) -> Result<(), IMStatusCode> {
        self.chunks.set(self.chunks.get() + 1);

        if let Some(attr_reports) = &report.attr_reports {
            let mut attrs = self.attrs.borrow_mut();

            for attr_resp in attr_reports.iter() {
                if let Ok(AttrResp::Data(data)) = attr_resp {
                    attrs.push((
                        data.path.endpoint.unwrap_or(u16::MAX),
                        data.path.cluster.unwrap_or(u32::MAX),
                        data.path.attr.unwrap_or(u32::MAX),
                    ));
                }
            }
        }

        // Park the first chunk. The publisher is waiting for its status
        // response, with the quiescence guard held.
        self.gate.enter().await;

        if !report.more_chunks.unwrap_or(false) {
            self.done.notify();
        }

        Ok(())
    }
}

/// Drive a complete read of one attribute on the remote (tested) instance.
async fn read_once<C: Crypto>(
    im: &E2eRunner<C>,
    endpoint: u16,
    cluster: u32,
    attr: u32,
) -> Result<(), Error> {
    let exchange = im.initiate_exchange().await?;
    let mut sender = exchange.read_sender().await?;

    let path = AttrPath::from_gp(&GenericPath::new(Some(endpoint), Some(cluster), Some(attr)));
    let paths = [path];

    let mut chunk = loop {
        match sender.tx().await? {
            TxOutcome::BuildRequest(builder) => {
                sender = builder
                    .attr_requests_from(&paths)?
                    .fabric_filtered(false)?
                    .end()?;
            }
            TxOutcome::GotResponse(chunk) => break chunk,
        }
    };

    loop {
        let _ = chunk.response()?;

        match chunk.complete().await? {
            Some(next) => chunk = next,
            None => break,
        }
    }

    Ok(())
}

/// Drive a complete write of one attribute on the remote (tested) instance.
async fn write_once<C: Crypto>(
    im: &E2eRunner<C>,
    endpoint: u16,
    cluster: u32,
    attr: u32,
) -> Result<(), Error> {
    let exchange = im.initiate_exchange().await?;
    let mut sender = exchange.write_sender(None).await?;

    let value_tlv = [0x05, 0x39, 0x05];
    let value = TLVElement::new(&value_tlv);

    let handle = loop {
        match sender.tx().await? {
            TxOutcome::BuildRequest(builder) => {
                let entries = builder.write_requests()?;

                let entry = entries
                    .push()?
                    .path(endpoint, cluster, attr)?
                    .data(|w| value.to_tlv(&TLVTag::Context(AttrDataTag::Data as u8), w))?
                    .end()?;

                sender = entry.end()?.end()?;
            }
            TxOutcome::GotResponse(handle) => break handle,
        }
    };

    let resp = handle.response()?;

    for status in resp.write_responses.iter() {
        let status = status?;
        assert_eq!(status.status.status, IMStatusCode::Success);
    }

    Ok(())
}

/// Drive a complete invoke of one command on the remote (tested) instance.
async fn invoke_once<C: Crypto>(
    im: &E2eRunner<C>,
    endpoint: u16,
    cluster: u32,
    cmd: u32,
) -> Result<(), Error> {
    let exchange = im.initiate_exchange().await?;
    let mut sender = exchange.invoke_sender(None).await?;

    let mut chunk = loop {
        match sender.tx().await? {
            TxOutcome::BuildRequest(builder) => {
                sender = builder
                    .suppress_response(false)?
                    .timed_request(false)?
                    .invoke_requests()?
                    .push()?
                    .path(endpoint, cluster, cmd)?
                    .data(|w| w.u8(&TLVTag::Context(CmdDataTag::Data as u8), 5))?
                    .end()?
                    .end()?
                    .end()?;
            }
            TxOutcome::GotResponse(chunk) => break chunk,
        }
    };

    loop {
        let _ = chunk.response()?;

        match chunk.complete().await? {
            Some(next) => chunk = next,
            None => break,
        }
    }

    Ok(())
}

/// Establish a subscription for `path` on the remote (tested) instance and
/// return its subscription id.
async fn subscribe_once<C: Crypto>(im: &E2eRunner<C>, path: AttrPath) -> Result<u32, Error> {
    let exchange = im.initiate_exchange().await?;
    let mut sender = exchange.subscribe_sender().await?;

    let paths = [path];

    let mut chunk = loop {
        match sender.tx().await? {
            TxOutcome::BuildRequest(builder) => {
                sender = builder
                    .keep_subs(true)?
                    .min_int_floor(0)?
                    .max_int_ceil(60)?
                    .attr_requests_from(&paths)?
                    .fabric_filtered(false)?
                    .end()?;
            }
            TxOutcome::GotResponse(chunk) => break chunk,
        }
    };

    let established = loop {
        let _ = chunk.response()?;

        match chunk.complete().await? {
            SubscribeOutcome::NextChunk(next) => chunk = next,
            SubscribeOutcome::Established(established) => break established,
        }
    };

    Ok(established.subscription_id)
}

/// A read parked inside its handler is in flight; once it completes - while
/// the server is still being polled - the model is idle again.
#[test]
fn read_await_gap_is_busy_and_releases_idle() {
    with_large_stack(|| {
        init_env_logger();

        let im = new_default_runner();
        im.add_default_acl();

        let gate = Gate::new(true);
        let trigger = Notification::new();
        let handler = GatedHandler::new(im.handler(), &gate, &trigger);
        let quiescence = im.state.quiescence();

        block_on(async {
            let server = im.run(handler);

            let driver = async {
                let client = async {
                    read_once(
                        &im,
                        1,
                        echo_cluster::ID,
                        echo_cluster::AttributesDiscriminants::Att1 as u32,
                    )
                    .await
                };

                let observer = async {
                    gate.wait_entered().await;

                    assert!(
                        !quiescence.is_idle(),
                        "a read parked in its handler must keep the model busy"
                    );

                    gate.release();
                };

                let (client_result, ()) = join(client, observer).await;
                client_result?;

                quiescence.wait_idle().await;

                assert!(
                    quiescence.is_idle(),
                    "after the read completes the model must be idle"
                );

                Ok::<_, Error>(())
            };

            select(server, driver).coalesce().await.unwrap();

            Ok::<_, Error>(())
        })
        .unwrap();
    });
}

/// A new request is served while another one is parked inside its handler -
/// the quiescence gate must not stop the responder from accepting new work.
#[test]
fn new_request_is_served_while_another_is_blocked() {
    with_large_stack(|| {
        init_env_logger();

        let im = new_default_runner();
        im.add_default_acl();

        let gate = Gate::new(true);
        let trigger = Notification::new();
        let handler = GatedHandler::new(im.handler(), &gate, &trigger);
        let quiescence = im.state.quiescence();

        block_on(async {
            let server = im.run(handler);

            let driver = async {
                let first = async {
                    read_once(
                        &im,
                        1,
                        echo_cluster::ID,
                        echo_cluster::AttributesDiscriminants::Att1 as u32,
                    )
                    .await
                };

                let second = async {
                    gate.wait_entered().await;

                    assert!(
                        !quiescence.is_idle(),
                        "the first read must still keep the model busy"
                    );

                    // A different attribute is not gated, so this read must
                    // complete while the first one is still parked.
                    read_once(
                        &im,
                        1,
                        echo_cluster::ID,
                        echo_cluster::AttributesDiscriminants::Att2 as u32,
                    )
                    .await?;

                    gate.release();

                    Ok::<_, Error>(())
                };

                let (first_result, second_result) = join(first, second).await;
                first_result?;
                second_result?;

                quiescence.wait_idle().await;

                assert!(quiescence.is_idle());

                Ok::<_, Error>(())
            };

            select(server, driver).coalesce().await.unwrap();

            Ok::<_, Error>(())
        })
        .unwrap();
    });
}

/// A write parked inside its handler is in flight; once it completes - while
/// the server is still being polled - the model is idle again.
#[test]
fn write_await_gap_is_busy_and_releases_idle() {
    with_large_stack(|| {
        init_env_logger();

        let im = new_default_runner();
        im.add_default_acl();

        let gate = Gate::new(true);
        let trigger = Notification::new();
        let handler = GatedHandler::new(im.handler(), &gate, &trigger);
        let quiescence = im.state.quiescence();

        block_on(async {
            let server = im.run(handler);

            let driver = async {
                let client = async {
                    write_once(
                        &im,
                        1,
                        echo_cluster::ID,
                        echo_cluster::AttributesDiscriminants::AttWrite as u32,
                    )
                    .await
                };

                let observer = async {
                    gate.wait_entered().await;

                    assert!(
                        !quiescence.is_idle(),
                        "a write parked in its handler must keep the model busy"
                    );

                    gate.release();
                };

                let (client_result, ()) = join(client, observer).await;
                client_result?;

                quiescence.wait_idle().await;

                assert!(
                    quiescence.is_idle(),
                    "after the write completes the model must be idle"
                );

                Ok::<_, Error>(())
            };

            select(server, driver).coalesce().await.unwrap();

            Ok::<_, Error>(())
        })
        .unwrap();
    });
}

/// An invoke parked inside its handler is in flight; once it completes - while
/// the server is still being polled - the model is idle again.
#[test]
fn invoke_await_gap_is_busy_and_releases_idle() {
    with_large_stack(|| {
        init_env_logger();

        let im = new_default_runner();
        im.add_default_acl();

        let gate = Gate::new(true);
        let trigger = Notification::new();
        let handler = GatedHandler::new(im.handler(), &gate, &trigger);
        let quiescence = im.state.quiescence();

        block_on(async {
            let server = im.run(handler);

            let driver = async {
                let client = async {
                    invoke_once(
                        &im,
                        1,
                        echo_cluster::ID,
                        echo_cluster::Commands::EchoReq as u32,
                    )
                    .await
                };

                let observer = async {
                    gate.wait_entered().await;

                    assert!(
                        !quiescence.is_idle(),
                        "an invoke parked in its handler must keep the model busy"
                    );

                    gate.release();
                };

                let (client_result, ()) = join(client, observer).await;
                client_result?;

                quiescence.wait_idle().await;

                assert!(
                    quiescence.is_idle(),
                    "after the invoke completes the model must be idle"
                );

                Ok::<_, Error>(())
            };

            select(server, driver).coalesce().await.unwrap();

            Ok::<_, Error>(())
        })
        .unwrap();
    });
}

/// The subscribe request and its priming report are in flight until the
/// subscribe exchange completes. Once established - while the server is still
/// being polled - the subscription and the session survive and further
/// requests succeed, and the model is idle.
#[test]
fn subscribe_priming_gap_is_busy_and_established_subscription_is_idle() {
    with_large_stack(|| {
        init_env_logger();

        let im = new_default_runner();
        im.add_default_acl();

        let gate = Gate::new(true);
        let trigger = Notification::new();
        let handler = GatedHandler::new(im.handler(), &gate, &trigger);
        let quiescence = im.state.quiescence();

        block_on(async {
            let server = im.run(handler);

            let driver = async {
                let client = async {
                    let path = AttrPath::from_gp(&GenericPath::new(
                        Some(1),
                        Some(echo_cluster::ID),
                        Some(echo_cluster::AttributesDiscriminants::Att1 as u32),
                    ));

                    subscribe_once(&im, path).await
                };

                let observer = async {
                    gate.wait_entered().await;

                    assert!(
                        !quiescence.is_idle(),
                        "the priming report must keep the model busy"
                    );

                    gate.release();
                };

                let (subscription_id, ()) = join(client, observer).await;
                let subscription_id = subscription_id?;
                assert_ne!(subscription_id, 0);

                quiescence.wait_idle().await;

                assert!(
                    quiescence.is_idle(),
                    "an established subscription must not keep the model busy"
                );

                // The server kept the subscription and the client session, and
                // keeps serving new requests.
                assert!(
                    im.state
                        .subscriptions()
                        .load_stats(None)
                        .current_subscriptions
                        > 0,
                    "the established subscription must survive"
                );
                assert!(
                    im.matter_client()
                        .with_state(|state| state.sessions.iter().count())
                        > 0,
                    "the client session must survive"
                );

                read_once(
                    &im,
                    1,
                    echo_cluster::ID,
                    echo_cluster::AttributesDiscriminants::Att2 as u32,
                )
                .await?;

                quiescence.wait_idle().await;
                assert!(quiescence.is_idle());

                Ok::<_, Error>(())
            };

            select(server, driver).coalesce().await.unwrap();

            Ok::<_, Error>(())
        })
        .unwrap();
    });
}

/// An outbound report parked between chunks is in flight; the same report
/// action carries both the changed fixed attributes and the bumped
/// `ConfigurationVersion`; and completing the report - while the server is
/// still being polled - returns the model to idle.
#[test]
fn chunked_outbound_report_is_busy_between_chunks_and_carries_shape_and_version_together() {
    with_large_stack(|| {
        init_env_logger();

        let im = new_default_runner();
        im.add_default_acl();

        let handler_gate = Gate::new(false);
        let trigger = Notification::new();
        let report_gate = Gate::new(true);

        let ctrl_node = Node::new(&[]);
        let ctrl_handler = (ctrl_node, EmptyHandler);
        let ctrl_buffers: MatterBuffers = MatterBuffers::new();
        let ctrl_state: InteractionModelState<DummyNetworks, 3, E2E_EVENTS_BUF_SIZE> =
            InteractionModelState::new(DummyNetworks);
        let reports = GatedReports::new(&report_gate);
        let ctrl_kv = im.matter_client().kv(DummyKvBlobStore);
        let ctrl_dm = InteractionModel::new_with_reports(
            im.matter_client(),
            rs_matter::crypto::test_only_crypto(),
            &ctrl_buffers,
            ctrl_handler,
            &ctrl_kv,
            NoopWirelessNetCtl::new(NetworkType::Ethernet),
            &reports,
            &ctrl_state,
        );
        let ctrl_responder = Responder::new_default(&ctrl_dm);

        let quiescence = im.state.quiescence();

        block_on(async {
            let server = im.run(GatedHandler::new(im.handler(), &handler_gate, &trigger));
            let controller = ctrl_responder.run::<4>();
            let ctrl_run = ctrl_dm.run();

            let client = async {
                // Subscribe to every attribute so the incremental report is
                // large enough to chunk.
                let path = AttrPath::from_gp(&GenericPath::new(None, None, None));
                let subscription_id = subscribe_once(&im, path).await?;
                assert_ne!(subscription_id, 0);

                quiescence.wait_idle().await;

                assert!(
                    quiescence.is_idle(),
                    "an established subscription must not keep the model busy"
                );

                // The handler marks every cluster dirty and bumps the
                // configuration version in one synchronous commit.
                trigger.notify();

                report_gate.wait_entered().await;

                assert!(
                    !quiescence.is_idle(),
                    "an outbound report waiting for its chunk acknowledgement must keep the model busy"
                );

                report_gate.release();

                reports.done.wait().await;
                quiescence.wait_idle().await;

                assert!(
                    quiescence.is_idle(),
                    "after the report completes the model must be idle"
                );

                assert!(
                    reports.chunks.get() > 1,
                    "the wildcard report must chunk (got {} chunk(s))",
                    reports.chunks.get()
                );

                let attrs = reports.attrs.borrow();

                assert!(
                    attrs.contains(&(
                        1,
                        echo_cluster::ID,
                        echo_cluster::AttributesDiscriminants::Att1 as u32
                    )),
                    "the changed fixed attribute and the version must be in the same report action: {attrs:?}"
                );

                assert!(
                    attrs.contains(&(0, 0x28, 0x18)),
                    "ConfigurationVersion must be in the same report action: {attrs:?}"
                );

                Ok::<_, Error>(())
            };

            let first = select(server, controller).coalesce();
            let second = select(ctrl_run, client).coalesce();

            select(first, second).coalesce().await.unwrap();

            Ok::<_, Error>(())
        })
        .unwrap();
    });
}

/// Dropping the Interaction Model mid-interaction (cancellation) releases the
/// quiescence guard through RAII.
#[test]
fn cancelled_interaction_releases_quiescence() {
    with_large_stack(|| {
        init_env_logger();

        let im = new_default_runner();
        im.add_default_acl();

        let gate = Gate::new(true);
        let trigger = Notification::new();
        let handler = GatedHandler::new(im.handler(), &gate, &trigger);
        let quiescence = im.state.quiescence();

        block_on(async {
            let server = im.run(handler);

            let client = async {
                read_once(
                    &im,
                    1,
                    echo_cluster::ID,
                    echo_cluster::AttributesDiscriminants::Att1 as u32,
                )
                .await
            };

            let observer = async {
                gate.wait_entered().await;

                assert!(
                    !quiescence.is_idle(),
                    "the read must be in flight before cancellation"
                );

                // Return without releasing the gate: completing the select
                // drops the server mid-interaction.
            };

            select(select(server, client).coalesce(), observer).await;

            Ok::<_, Error>(())
        })
        .unwrap();

        assert!(
            quiescence.is_idle(),
            "cancelling an in-flight interaction must release the quiescence guard"
        );
    });
}
