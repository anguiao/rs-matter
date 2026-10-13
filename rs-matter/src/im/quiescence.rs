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

//! Tracking of whether the Interaction Model is quiescent - i.e. whether it
//! has any in-flight interaction or outbound report.
//!
//! A node's shape (`Node` metadata: endpoints, clusters, attributes, ...) is
//! traversed while an interaction or an outbound report is being generated and
//! sent. Changing that shape in the middle of such a traversal would expose a
//! torn view to the peer, so an application that wants to reconfigure the node
//! at runtime has to wait until the Interaction Model is idle.
//!
//! [`InteractionQuiescence`] is that boundary. The Interaction Model holds an
//! [`InteractionGuard`] for as long as a single interaction (read, write,
//! invoke, timed request, subscribe, priming report) or a single outbound
//! `ReportData` action (all of its chunks, through the final acknowledgement)
//! is in flight. An application can then:
//!
//! 1. [`InteractionQuiescence::wait_idle`] until the counter drops to zero;
//! 2. synchronously apply the new shape, bump `ConfigurationVersion` and
//!    notify every affected cluster - without awaiting in between.
//!
//! Because the executor is single-threaded, no interaction can start between
//! the instant `wait_idle` resolves and the end of that same poll, so the
//! shape change is applied atomically with respect to interaction traversals.
//!
//! Long-lived transports, established subscriptions and actions submitted to
//! an external executor are deliberately *not* in-flight interactions: they
//! do not hold a guard, so they do not prevent the node shape from changing.

use crate::utils::init::{init, Init};
use crate::utils::sync::Signal;

/// Tracks the number of in-flight Interaction Model interactions and outbound
/// reports.
///
/// See the [module docs](self) for the contract this type enforces.
///
/// Like the other single-waiter notifications in this crate, only one task
/// should await [`InteractionQuiescence::wait_idle`] on a given instance at a
/// time.
pub struct InteractionQuiescence {
    in_flight: Signal<usize>,
}

impl InteractionQuiescence {
    /// Create a new, idle instance.
    pub const fn new() -> Self {
        Self {
            in_flight: Signal::new(0),
        }
    }

    /// Return an in-place initializer for the instance.
    pub fn init() -> impl Init<Self> {
        init!(Self {
            in_flight <- Signal::init(0),
        })
    }

    /// Whether no interaction or outbound report is currently in flight.
    pub fn is_idle(&self) -> bool {
        self.in_flight.access(|in_flight| *in_flight == 0)
    }

    /// Wait until no interaction or outbound report is in flight.
    ///
    /// In a single-threaded executor the state cannot change between the
    /// moment this future resolves and the next suspension point, so an
    /// application may synchronously reconfigure the node immediately after
    /// awaiting this method - provided it does not await before committing
    /// the change.
    pub async fn wait_idle(&self) {
        self.in_flight
            .wait(|in_flight| (*in_flight == 0).then_some(()))
            .await
    }

    /// Mark one interaction or outbound report as in flight until the returned
    /// guard is dropped.
    ///
    /// The guard is the RAII protection of the [module contract](self): it is
    /// released on every exit path, including cancellation of the future that
    /// holds it.
    pub(crate) fn guard(&self) -> InteractionGuard<'_> {
        self.in_flight.modify(|in_flight| {
            // Plain arithmetic: the RAII pairing in this module is an internal
            // invariant, so a bug should surface as an overflow/underflow panic
            // rather than be silently absorbed.
            *in_flight += 1;

            (false, ())
        });

        InteractionGuard { quiescence: self }
    }
}

impl Default for InteractionQuiescence {
    fn default() -> Self {
        Self::new()
    }
}

/// An RAII guard marking one Interaction Model interaction or outbound report
/// as in flight.
///
/// Created by [`InteractionQuiescence::guard`]; releases the marking (and
/// wakes a [`InteractionQuiescence::wait_idle`] waiter once the count reaches
/// zero) when dropped.
pub struct InteractionGuard<'a> {
    quiescence: &'a InteractionQuiescence,
}

impl Drop for InteractionGuard<'_> {
    fn drop(&mut self) {
        self.quiescence.in_flight.modify(|in_flight| {
            // Plain arithmetic: see `guard()`.
            *in_flight -= 1;

            (*in_flight == 0, ())
        });
    }
}
