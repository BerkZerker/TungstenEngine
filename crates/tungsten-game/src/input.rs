use std::collections::HashSet;

use winit::keyboard::KeyCode;

/// Input gathered between frames.
#[derive(Default)]
pub struct Input {
    held: HashSet<KeyCode>,
    /// Keys that went down since the last frame.
    pub pressed: Vec<KeyCode>,
    /// Mouse motion since the last frame, in device units.
    pub look: (f64, f64),
    pub break_held: bool,
    /// Right clicks since the last tick.
    pub place_clicks: u32,
    /// Scroll since the last frame, in lines; positive is away from the user.
    pub scroll: f32,
}

/// Recognises two presses of a key in quick succession.
#[derive(Default)]
pub struct DoubleTap {
    last: Option<u64>,
}

impl DoubleTap {
    /// 0.3 s at 60 ticks a second.
    const WINDOW_TICKS: u64 = 18;

    /// Record a press at `tick`. True if it completes a double tap.
    pub fn press(&mut self, tick: u64) -> bool {
        let double = self.last.is_some_and(|t| tick - t <= Self::WINDOW_TICKS);
        // A third quick press starts over rather than toggling back.
        self.last = (!double).then_some(tick);
        double
    }
}

impl Input {
    pub fn key(&mut self, code: KeyCode, down: bool) {
        if down {
            if self.held.insert(code) {
                self.pressed.push(code);
            }
        } else {
            self.held.remove(&code);
        }
    }

    pub fn held(&self, code: KeyCode) -> bool {
        self.held.contains(&code)
    }

    pub fn any_held(&self, codes: &[KeyCode]) -> bool {
        codes.iter().any(|c| self.held.contains(c))
    }

    /// -1, 0 or 1 from a pair of opposing keys.
    pub fn axis(&self, negative: KeyCode, positive: KeyCode) -> f32 {
        self.held(positive) as i32 as f32 - self.held(negative) as i32 as f32
    }

    /// Forget everything held, e.g. when the window loses focus.
    pub fn release_all(&mut self) {
        self.held.clear();
        self.break_held = false;
    }

    pub fn end_frame(&mut self) {
        self.pressed.clear();
        self.look = (0.0, 0.0);
        self.scroll = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_tap_needs_two_quick_presses() {
        let mut d = DoubleTap::default();
        assert!(!d.press(100));
        assert!(!d.press(119), "too slow");
        assert!(d.press(137));
        // The pair is spent: a third press is a first press again.
        assert!(!d.press(140));
        assert!(d.press(150));
    }
}
