//! A small priority thread pool. Lower keys run first; equal keys run in
//! submission order.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

struct Item<J> {
    key: u64,
    seq: u64,
    job: J,
}

impl<J> PartialEq for Item<J> {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key && self.seq == o.seq
    }
}
impl<J> Eq for Item<J> {}
impl<J> PartialOrd for Item<J> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl<J> Ord for Item<J> {
    fn cmp(&self, o: &Self) -> Ordering {
        // BinaryHeap is a max-heap; invert so the smallest key pops first.
        (o.key, o.seq).cmp(&(self.key, self.seq))
    }
}

struct State<J> {
    heap: BinaryHeap<Item<J>>,
    seq: u64,
    running: usize,
    shutdown: bool,
}

struct Shared<J> {
    state: Mutex<State<J>>,
    work: Condvar,
}

pub struct Pool<J: Send + 'static> {
    shared: Arc<Shared<J>>,
    threads: Vec<JoinHandle<()>>,
}

impl<J: Send + 'static> Pool<J> {
    pub fn new(threads: usize, run: impl Fn(J) + Send + Sync + 'static) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { heap: BinaryHeap::new(), seq: 0, running: 0, shutdown: false }),
            work: Condvar::new(),
        });
        let run = Arc::new(run);
        let threads = (0..threads.max(1))
            .map(|i| {
                let (shared, run) = (shared.clone(), run.clone());
                std::thread::Builder::new()
                    .name(format!("tungsten-worker-{i}"))
                    .spawn(move || {
                        loop {
                            let job = {
                                let mut st = shared.state.lock().unwrap();
                                loop {
                                    if st.shutdown {
                                        return;
                                    }
                                    if let Some(item) = st.heap.pop() {
                                        st.running += 1;
                                        break item.job;
                                    }
                                    st = shared.work.wait(st).unwrap();
                                }
                            };
                            run(job);
                            shared.state.lock().unwrap().running -= 1;
                        }
                    })
                    .expect("spawn worker")
            })
            .collect();
        Self { shared, threads }
    }

    pub fn push(&self, key: u64, job: J) {
        let mut st = self.shared.state.lock().unwrap();
        let seq = st.seq;
        st.seq += 1;
        st.heap.push(Item { key, seq, job });
        drop(st);
        self.shared.work.notify_one();
    }

    /// Queue many jobs under one lock.
    pub fn push_all(&self, jobs: impl IntoIterator<Item = (u64, J)>) {
        let mut st = self.shared.state.lock().unwrap();
        for (key, job) in jobs {
            let seq = st.seq;
            st.seq += 1;
            st.heap.push(Item { key, seq, job });
        }
        drop(st);
        self.shared.work.notify_all();
    }

    /// Re-prioritise queued jobs. `f` returns the new key, or `None` to drop the job.
    pub fn rekey(&self, mut f: impl FnMut(&J) -> Option<u64>) {
        let mut st = self.shared.state.lock().unwrap();
        let items: Vec<Item<J>> = std::mem::take(&mut st.heap).into_vec();
        st.heap = items
            .into_iter()
            .filter_map(|mut it| {
                it.key = f(&it.job)?;
                Some(it)
            })
            .collect();
    }

    /// Jobs queued and jobs currently running.
    pub fn load(&self) -> (usize, usize) {
        let st = self.shared.state.lock().unwrap();
        (st.heap.len(), st.running)
    }

    pub fn threads(&self) -> usize {
        self.threads.len()
    }
}

impl<J: Send + 'static> Drop for Pool<J> {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().shutdown = true;
        self.shared.work.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn runs_in_key_order_and_rekeys() {
        let (tx, rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let g = gate.clone();
        let tx = Mutex::new(tx);
        let pool = Pool::new(1, move |j: u32| {
            if j == 0 {
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
            }
            tx.lock().unwrap().send(j).unwrap();
        });
        pool.push(0, 0);
        // Wait until the blocker is running so the rest stay queued.
        while pool.load().1 == 0 {
            std::thread::yield_now();
        }
        for j in [5u32, 3, 9, 4] {
            pool.push(j as u64, j);
        }
        pool.rekey(|&j| if j == 9 { None } else { Some(100 - j as u64) });
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let got: Vec<u32> = (0..4).map(|_| rx.recv().unwrap()).collect();
        assert_eq!(got, vec![0, 5, 4, 3]);
    }
}
