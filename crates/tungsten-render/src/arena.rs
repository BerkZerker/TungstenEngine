//! Sub-allocation of many small meshes out of a few large GPU buffers.

/// A range inside one page of an [`Arena`], in elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Alloc {
    pub page: u32,
    pub offset: u32,
    pub len: u32,
}

struct Page {
    buffer: wgpu::Buffer,
    /// Free ranges `(offset, len)` in elements, sorted by offset, never adjacent.
    free: Vec<(u32, u32)>,
}

pub struct Arena {
    label: &'static str,
    usage: wgpu::BufferUsages,
    /// Bytes per element.
    stride: u64,
    /// Elements per page.
    page_len: u32,
    pages: Vec<Page>,
    used: u64,
}

impl Arena {
    pub fn new(label: &'static str, usage: wgpu::BufferUsages, stride: u64, page_len: u32) -> Self {
        Self { label, usage: usage | wgpu::BufferUsages::COPY_DST, stride, page_len, pages: Vec::new(), used: 0 }
    }

    pub fn buffer(&self, page: u32) -> &wgpu::Buffer {
        &self.pages[page as usize].buffer
    }

    pub fn used_bytes(&self) -> u64 {
        self.used * self.stride
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.pages.len() as u64 * self.page_len as u64 * self.stride
    }

    fn take(free: &mut Vec<(u32, u32)>, len: u32) -> Option<u32> {
        let i = free.iter().position(|&(_, l)| l >= len)?;
        let (offset, l) = free[i];
        if l == len {
            free.remove(i);
        } else {
            free[i] = (offset + len, l - len);
        }
        Some(offset)
    }

    pub fn alloc(&mut self, device: &wgpu::Device, len: u32) -> Alloc {
        assert!(len > 0 && len <= self.page_len, "{}: allocation of {len} elements", self.label);
        self.used += len as u64;
        for (p, page) in self.pages.iter_mut().enumerate() {
            if let Some(offset) = Self::take(&mut page.free, len) {
                return Alloc { page: p as u32, offset, len };
            }
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: self.page_len as u64 * self.stride,
            usage: self.usage,
            mapped_at_creation: false,
        });
        let mut free = vec![(0, self.page_len)];
        let offset = Self::take(&mut free, len).unwrap();
        self.pages.push(Page { buffer, free });
        Alloc { page: self.pages.len() as u32 - 1, offset, len }
    }

    pub fn free(&mut self, a: Alloc) {
        self.used -= a.len as u64;
        let free = &mut self.pages[a.page as usize].free;
        let i = free.partition_point(|&(o, _)| o < a.offset);
        free.insert(i, (a.offset, a.len));
        // Merge with the following and preceding ranges.
        if i + 1 < free.len() && free[i].0 + free[i].1 == free[i + 1].0 {
            free[i].1 += free[i + 1].1;
            free.remove(i + 1);
        }
        if i > 0 && free[i - 1].0 + free[i - 1].1 == free[i].0 {
            free[i - 1].1 += free[i].1;
            free.remove(i);
        }
    }

    pub fn write(&self, queue: &wgpu::Queue, a: Alloc, bytes: &[u8]) {
        debug_assert!(bytes.len() as u64 <= a.len as u64 * self.stride);
        queue.write_buffer(&self.pages[a.page as usize].buffer, a.offset as u64 * self.stride, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::Arena;

    #[test]
    fn free_list_splits_and_merges() {
        let mut free = vec![(0u32, 100u32)];
        let a = Arena::take(&mut free, 30).unwrap();
        let b = Arena::take(&mut free, 30).unwrap();
        let c = Arena::take(&mut free, 40).unwrap();
        assert_eq!((a, b, c), (0, 30, 60));
        assert!(free.is_empty() && Arena::take(&mut free, 1).is_none());
    }
}
