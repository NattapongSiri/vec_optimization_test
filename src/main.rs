use std::time::*;
use std::borrow::Borrow;
/// A pair of vecs. One to store keys. Another is to store values.
/// 
/// The problem with this approach is about keeping the two vecs completely
/// sync. It more error prone. To make it less erratic, every operation that add
/// or remove element(s) to keys or values shall have assert.
/// 
/// Another problem with this approach is about slice operation. Since keys and
/// values is completely separate, it's hard to do slicing on this struct.
#[derive(Clone)]
struct VecPair<K, V> {
    keys: Vec<K>,
    values: Vec<V>
}

impl<K, V> VecPair<K, V> {
    pub fn new() -> Self {
        VecPair {
            keys: Vec::new(),
            values: Vec::new()
        }
    }

    pub fn with_capacity(size: usize) -> Self {
        VecPair {
            keys: Vec::with_capacity(size),
            values: Vec::with_capacity(size)
        }
    }

    pub fn push(&mut self, key: K, value: V) {
        self.keys.push(key);
        self.values.push(value);
        debug_assert_eq!(self.keys.len(), self.values.len());
    }

    pub fn pop(&mut self) -> Option<(K, V)> {
        let result = self.keys.pop().and_then(|key| {Some((key, self.values.pop().unwrap()))});
        debug_assert_eq!(self.keys.len(), self.values.len());
        result
    }

    pub fn insert(&mut self, index: usize, key: K, value: V) {
        self.keys.insert(index, key);
        self.values.insert(index, value);
        debug_assert_eq!(self.keys.len(), self.values.len());
    }

    pub fn remove(&mut self, index: usize) -> (K, V) {
        let key = self.keys.remove(index);
        let value = self.values.remove(index);
        debug_assert_eq!(self.keys.len(), self.values.len());
        (key, value)
    }

    pub fn iter(&self) -> core::iter::Zip<core::slice::Iter<K>, core::slice::Iter<V>> {
        self.keys.iter().zip(self.values.iter())
    }

    pub fn iter_mut(&mut self) -> core::iter::Zip<core::slice::IterMut<K>, core::slice::IterMut<V>> {
        self.keys.iter_mut().zip(self.values.iter_mut())
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.keys.len(), self.values.len());
        self.keys.len()
    }
}

impl<K, V> IntoIterator for VecPair<K, V> {
    type IntoIter = core::iter::Zip<std::vec::IntoIter<K>, std::vec::IntoIter<V>>;
    type Item = (K, V);

    fn into_iter(self) -> Self::IntoIter {
        self.keys.into_iter().zip(self.values.into_iter())
    }
}

impl<'a, K, V> IntoIterator for &'a VecPair<K, V> {
    type IntoIter = core::iter::Zip<core::slice::Iter<'a, K>, core::slice::Iter<'a, V>>;
    type Item = (&'a K, &'a V);

    fn into_iter(self) -> core::iter::Zip<core::slice::Iter<'a, K>, core::slice::Iter<'a, V>> {
        self.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut VecPair<K, V> {
    type IntoIter = core::iter::Zip<core::slice::IterMut<'a, K>, core::slice::IterMut<'a, V>>;
    type Item = (&'a mut K, &'a mut V);

    fn into_iter(self) -> core::iter::Zip<core::slice::IterMut<'a, K>, core::slice::IterMut<'a, V>> {
        self.iter_mut()
    }
}

/// A vec of tuple of key and value.
/// 
/// The problem with this approach is the efficiency. If user need to frequently lookup
/// for key to retrieve a value, it is inefficient to iterate over vec. This is because on each
/// iteration, it need to read both key and value into stack and only the matched key will use
/// value. All the rest will not touch the value.
#[derive(Clone)]
struct TupleVec<K, V> {
    entries: Vec<(K, V)>
}

impl<K, V> TupleVec<K, V> {
    pub fn new() -> Self {
        TupleVec {
            entries: Vec::new()
        }
    }

    pub fn with_capacity(size: usize) -> Self {
        TupleVec {
            entries: Vec::with_capacity(size),
        }
    }
    pub fn push(&mut self, key: K, value: V) {
        self.entries.push((key, value));
    }

    pub fn pop(&mut self) -> Option<(K, V)> {
        self.entries.pop()
    }

    pub fn insert(&mut self, index: usize, key: K, value: V) {
        self.entries.insert(index, (key, value));
    }

    pub fn remove(&mut self, index: usize) -> (K, V) {
        self.entries.remove(index)
    }

    pub fn iter(&self) -> core::slice::Iter<(K, V)> {
        self.entries.iter()
    }

    pub fn iter_mut(&mut self) -> core::slice::IterMut<(K, V)> {
        self.entries.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<K, V> IntoIterator for TupleVec<K, V> {
    type IntoIter = std::vec::IntoIter<(K, V)>;
    type Item = (K, V);

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a TupleVec<K, V> {
    type IntoIter = core::slice::Iter<'a, (K, V)>;
    type Item = &'a (K, V);

    fn into_iter(self) -> core::slice::Iter<'a, (K, V)> {
        self.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut TupleVec<K, V> {
    type IntoIter = core::slice::IterMut<'a, (K, V)>;
    type Item = &'a mut (K, V);

    fn into_iter(self) -> core::slice::IterMut<'a, (K, V)> {
        self.iter_mut()
    }
}

/// A vec where each keys itself is a flatten key where each key is a Box[T] or Vec<T>.
/// It should not have any benefit over Vec<Box[T]> or Vec<Vec<T>>.
/// It may a bit slower instead because on each access, it need to calculate 
/// the first index of key and the length of that key
#[derive(Clone, Debug)]
struct FlattenVec<T> {
    values: Vec<T>,
    indices: Vec<usize>
}

impl<T> FlattenVec<T> {
    pub fn new() -> Self {
        FlattenVec {
            values: Vec::new(),
            indices: Vec::new()
        }
    }

    pub fn with_capacity(size: usize) -> Self {
        FlattenVec {
            values: Vec::with_capacity(size),
            indices: Vec::with_capacity(size)
        }
    }
    pub fn push(&mut self, value: Box<[T]>) {
        if self.values.len() > 0 {
            self.indices.push(self.values.len());
        }
        value.into_vec().into_iter().for_each(|v| {self.values.push(v)});
    }

    pub fn pop(&mut self) -> Option<Vec<T>> {
        if let Some(idx) = self.indices.pop() {
            Some(self.values.split_off(idx))
        } else {
            if self.values.len() == 0 {
                None
            } else {
                Some(self.values.split_off(0))
            }
        }
    }

    pub fn insert(&mut self, index: usize, value: Box<[T]>) {
        if index == 0 {
            self.indices.insert(0, value.len());
            self.indices.iter_mut().for_each(|i| {
                *i += value.len()
            });
            value.into_vec().into_iter().enumerate().for_each(|(i, v)| {
                self.values.insert(i, v);
            });
        } else if index < self.indices.len() {
            self.indices.insert(index, self.indices[index]);
            self.indices[(index + 1)..].iter_mut().for_each(|i| {
                *i += value.len()
            });
            value.into_vec().into_iter().fold(self.indices[index], |i, v| {
                self.values.insert(i, v);
                i + 1
            });
        } else {
            self.indices.push(self.values.len());
            self.values.append(&mut value.into_vec());
        }
    }

    pub fn remove(&mut self, index: usize) -> Vec<T> {
        let i = self.indices.remove(index);
        let j = if index < self.indices.len() {
            self.indices[index]
        } else {
            self.values.len()
        };
        let removal_len = j - i;
        self.indices[index..].iter_mut().for_each(|i| {*i -= removal_len});
        self.values.drain(i..j).collect()
    }

    pub fn iter(&self) -> FlattenVecIter<'_, T> {
        FlattenVecIter {
            index: 0,
            vecs: self
        }
    }

    pub fn iter_mut(&mut self) -> FlattenVecIterMut<'_, T> {
        FlattenVecIterMut {
            index: 0,
            vecs: self
        }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }
}

pub struct FlattenVecIter<'a, T> {
    index: usize,
    vecs: &'a FlattenVec<T>
}

impl<'a, T> Iterator for FlattenVecIter<'a, T> {
    type Item=&'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.vecs.indices.len() {
            return None
        }

        let start = if self.index > 0 {
            self.vecs.indices[self.index - 1]
        } else {
            0
        };
        let end = if self.index < self.vecs.indices.len() {
            self.vecs.indices[self.index]
        } else {
            self.vecs.values.len()
        };
        self.index += 1;
        Some(&self.vecs.values[start..end])
    }
}

impl<'a, T> ExactSizeIterator for FlattenVecIter<'a, T> {
    fn len(&self) -> usize {
        self.vecs.indices.len()
    }
}

pub struct FlattenVecIterMut<'a, T> {
    index: usize,
    vecs: &'a mut FlattenVec<T>
}

impl<'a, T> Iterator for FlattenVecIterMut<'a, T> {
    type Item=&'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.vecs.indices.len() {
            return None
        }

        let start = if self.index > 0 {
            self.vecs.indices[self.index - 1]
        } else {
            0
        };
        let end = if self.index < self.vecs.indices.len() {
            self.vecs.indices[self.index]
        } else {
            self.vecs.values.len()
        };
        self.index += 1;
        unsafe {
            Some(core::slice::from_raw_parts_mut(self.vecs.values.as_mut_ptr().add(start), end - start))
        }
    }
}

impl<'a, T> ExactSizeIterator for FlattenVecIterMut<'a, T> {
    fn len(&self) -> usize {
        self.vecs.indices.len()
    }
}

impl<'a, T> core::iter::FusedIterator for FlattenVecIterMut<'a, T> {}

pub struct FlattenVecIntoIter<T> {
    offset: usize,
    vecs: FlattenVec<T>
}

impl<T> Iterator for FlattenVecIntoIter<T> {
    type Item=Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.vecs.values.len() == 0 {
            return None
        }

        if self.vecs.indices.len() > 0 {
            let i = self.vecs.indices.remove(0);
            let mut result = self.vecs.values.split_off(i - self.offset);
            self.offset = i;
            core::mem::swap(&mut self.vecs.values, &mut result);
            Some(result)
        } else {
            Some(self.vecs.values.split_off(0))
        }
    }
}

impl<T> ExactSizeIterator for FlattenVecIntoIter<T> {
    fn len(&self) -> usize {
        self.vecs.len()
    }
}

impl<T> core::iter::FusedIterator for FlattenVecIntoIter<T> {}

impl<T> IntoIterator for FlattenVec<T> {
    type IntoIter = FlattenVecIntoIter<T>;
    type Item = Vec<T>;

    fn into_iter(self) -> Self::IntoIter {
        FlattenVecIntoIter {
            offset: 0, 
            vecs: self
        }
    }
}

/// To compaare FlattenVec with nested Vec, we need this type.
type NestedVec<T> = Vec<Box<[T]>>;

#[inline]
fn borrow_eq<T>(value: T, comparator: usize) -> bool where T: Borrow<usize> + PartialEq {
    *value.borrow() == comparator
}

macro_rules! bench_kv {
    ($e: ident, $max: expr, $s: expr) => {
        let mut temp = Vec::with_capacity($s * 3);
        let mut insts_times = Vec::with_capacity($s);
        let mut push_times = Vec::with_capacity($s);
        let mut iter_times = Vec::with_capacity($s);
        let mut iter_mut_times = Vec::with_capacity($s);
        let mut ins_front_times = Vec::with_capacity($s);
        let mut ins_back_times = Vec::with_capacity($s);
        let mut rm_front_times = Vec::with_capacity($s);
        let mut rm_back_times = Vec::with_capacity($s);
        let mut clone_times = Vec::with_capacity($s);
        let mut pop_times = Vec::with_capacity($s);
        let mut into_iter_times = Vec::with_capacity($s);
        print!("Bench {} for {} times.", stringify!($e), $s);

        for _ in 0..$s {
            print!(".");
            // println!("Begin benchmark {}", stringify!($e));
            let timer = Instant::now();
            let mut collection = $e::with_capacity($max + 2);
            let inst_time = timer.elapsed().as_nanos();
            insts_times.push(inst_time);
            // println!("Instantiated done in {}ns", inst_time);

            for i in 0..$max {
                collection.push(i, i);
            }
            let push_time = timer.elapsed().as_nanos() - inst_time;
            push_times.push(push_time);
            // println!("Push done in {}ms {}us {}ns", push_time / 1000000, (push_time / 1000) % 1000, push_time % 1000);
            
            let found = *collection.iter().find(|(k, v)| {
                borrow_eq(*k, $max - 1)
            }).unwrap().0.borrow();
            
            let iter_time = timer.elapsed().as_nanos() - push_time;
            iter_times.push(iter_time);
            // println!("Iterator done in {}ms {}us {}ns", iter_time / 1000000, (iter_time / 1000) % 1000, iter_time % 1000);

            
            collection.iter_mut().for_each(|(k, v)| {
                *k += 1;
                *v += 1;
            });
            
            let iter_mut_time = timer.elapsed().as_nanos() - iter_time;
            iter_mut_times.push(iter_mut_time);
            // println!("Iter mut done in {}ms {}us {}ns", iter_mut_time / 1000000, (iter_mut_time / 1000) % 1000, iter_mut_time % 1000);
            
            collection.insert(0, 0, 0);
            
            let insert_time = timer.elapsed().as_nanos() - iter_mut_time;
            ins_front_times.push(insert_time);
            // println!("Insert front done in {}ms {}us {}ns", insert_time / 1000000, (insert_time / 1000) % 1000, insert_time % 1000);
            
            collection.insert(collection.len(), 0, 0);
            
            let insert_back_time = timer.elapsed().as_nanos() - insert_time;
            ins_back_times.push(insert_back_time);
            // println!("Insert back done in {}ms {}us {}ns", insert_back_time / 1000000, (insert_back_time / 1000) % 1000, insert_back_time % 1000);
            
            collection.remove(0);
            
            let remove_time = timer.elapsed().as_nanos() - insert_back_time;
            rm_front_times.push(remove_time);
            // println!("Remove front done in {}ms {}us {}ns", remove_time / 1000000, (remove_time / 1000) % 1000, remove_time % 1000);
            
            collection.remove(collection.len() - 1);
            
            let remove_back_time = timer.elapsed().as_nanos() - remove_time;
            rm_back_times.push(remove_back_time);
            // println!("Remove back done in {}ms {}us {}ns", remove_back_time / 1000000, (remove_back_time / 1000) % 1000, remove_back_time % 1000);

            let mut cloned = collection.clone();
            let clone_time = timer.elapsed().as_nanos() - remove_back_time;
            clone_times.push(clone_time);
            // println!("Clone done in {}ms {}us {}ns", clone_time / 1000000, (clone_time / 1000) % 1000, clone_time % 1000);

            for _ in 0..$max {
                cloned.pop();
            }

            let pop_time = timer.elapsed().as_nanos() - clone_time;
            pop_times.push(pop_time);
            // println!("Pop done in {}ms {}us {}ns", pop_time / 1000000, (pop_time / 1000) % 1000, pop_time % 1000);

            let mut last = (0, 0);
            for entry in collection {
                if entry.0 == $max && entry.1 == $max {
                    last = entry
                }
            }

            let into_time = timer.elapsed().as_nanos() - pop_time;
            into_iter_times.push(into_time);
            // println!("IntoIter done in {}ms {}us {}ns", into_time / 1000000, (into_time / 1000) % 1000, into_time % 1000);
            
            temp.push(found);
            temp.push(last.0);
            temp.push(last.1);
        }

        // calculate mean for all measurement
        let insts_mean = insts_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let push_mean = push_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let iter_mean = iter_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let iter_mut_mean = iter_mut_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let ins_front_mean = ins_front_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let ins_back_mean = ins_back_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let rm_front_mean = rm_front_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let rm_back_mean = rm_back_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let clone_mean = clone_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let pop_mean = pop_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let into_iter_mean = into_iter_times.iter().fold(0, |a, t| a + t) / $s as u128;

        // calculate variance for all measurement
        let insts_var = insts_times.iter().fold(0, |a, t| a + (t - insts_mean).pow(2)) / $s as u128;
        let push_var = push_times.iter().fold(0, |a, t| a + (t - push_mean).pow(2)) / $s as u128;
        let iter_var = iter_times.iter().fold(0, |a, t| a + (t - iter_mean).pow(2)) / $s as u128;
        let iter_mut_var = iter_mut_times.iter().fold(0, |a, t| a + (t - iter_mut_mean).pow(2)) / $s as u128;
        let ins_front_var = ins_front_times.iter().fold(0, |a, t| a + (t - ins_front_mean).pow(2)) / $s as u128;
        let ins_back_var = ins_back_times.iter().fold(0, |a, t| a + (t - ins_back_mean).pow(2)) / $s as u128;
        let rm_front_var = rm_front_times.iter().fold(0, |a, t| a + (t - rm_front_mean).pow(2)) / $s as u128;
        let rm_back_var = rm_back_times.iter().fold(0, |a, t| a + (t - rm_back_mean).pow(2)) / $s as u128;
        let clone_var = clone_times.iter().fold(0, |a, t| a + (t - clone_mean).pow(2)) / $s as u128;
        let pop_var = pop_times.iter().fold(0, |a, t| a + (t - pop_mean).pow(2)) / $s as u128;
        let into_iter_var = into_iter_times.iter().fold(0, |a, t| a + (t - into_iter_mean).pow(2)) / $s as u128;

        println!();
        println!("Average instantiate time is {}ns with sd = {}", insts_mean, (insts_var as f64).powf(0.5));
        println!("Average push time is {}ms {}us {}ns with sd = {}", push_mean / 1_000_000, (push_mean / 1_000) % 1_000, push_mean % 1000, (push_var as f64).powf(0.5));
        println!("Average iter time is {}ms {}us {}ns with sd = {}", iter_mean / 1_000_000, (iter_mean / 1_000) % 1_000, iter_mean % 1000, (iter_var as f64).powf(0.5));
        println!("Average iter mut time is {}ms {}us {}ns with sd = {}", iter_mut_mean / 1_000_000, (iter_mut_mean / 1_000) % 1_000, iter_mut_mean % 1000, (iter_mut_var as f64).powf(0.5));
        println!("Average insert front time is {}ms {}us {}ns with sd = {}", ins_front_mean / 1_000_000, (ins_front_mean / 1_000) % 1_000, ins_front_mean % 1000, (ins_front_var as f64).powf(0.5));
        println!("Average insert back time is {}ms {}us {}ns with sd = {}", ins_back_mean / 1_000_000, (ins_back_mean / 1_000) % 1_000, ins_back_mean % 1000, (ins_back_var as f64).powf(0.5));
        println!("Average remove front time is {}ms {}us {}ns with sd = {}", rm_front_mean / 1_000_000, (rm_front_mean / 1_000) % 1_000, rm_front_mean % 1000, (rm_front_var as f64).powf(0.5));
        println!("Average remove back time is {}ms {}us {}ns with sd = {}", rm_back_mean / 1_000_000, (rm_back_mean / 1_000) % 1_000, rm_back_mean % 1000, (rm_back_var as f64).powf(0.5));
        println!("Average clone time is {}ms {}us {}ns with sd = {}", clone_mean / 1_000_000, (clone_mean / 1_000) % 1_000, clone_mean % 1000, (clone_var as f64).powf(0.5));
        println!("Average pop time is {}ms {}us {}ns with sd = {}", pop_mean / 1_000_000, (pop_mean / 1_000) % 1_000, pop_mean % 1000, (pop_var as f64).powf(0.5));
        println!("Average into iter time is {}ms {}us {}ns with sd = {}", into_iter_mean / 1_000_000, (into_iter_mean / 1_000) % 1_000, into_iter_mean % 1000, (into_iter_var as f64).powf(0.5));
        temp.into_iter().fold(0, |a, i| a + i);
    };
}

#[inline]
fn slice_eq<D, T>(slice: &D, comparison: &[T]) -> bool where D: AsRef<[T]>, T: PartialEq {
    slice.as_ref() == comparison
}

macro_rules! bench_nested {
    ($e: ident, $max: expr, $n:expr, $s: expr) => {
        let mut temp = Vec::with_capacity($s * 3);
        let mut insts_times = Vec::with_capacity($s);
        let mut push_times = Vec::with_capacity($s);
        let mut iter_times = Vec::with_capacity($s);
        let mut iter_mut_times = Vec::with_capacity($s);
        let mut ins_front_times = Vec::with_capacity($s);
        let mut ins_back_times = Vec::with_capacity($s);
        let mut rm_front_times = Vec::with_capacity($s);
        let mut rm_back_times = Vec::with_capacity($s);
        let mut clone_times = Vec::with_capacity($s);
        let mut pop_times = Vec::with_capacity($s);
        let mut into_iter_times = Vec::with_capacity($s);
        print!("Bench {} for {} times.", stringify!($e), $s);

        for _ in 0..$s {
            print!(".");
            // println!("Begin benchmark {}", stringify!($e));
            let timer = Instant::now();
            let mut collection = $e::with_capacity($max + 2);
            let inst_time = timer.elapsed().as_nanos();
            insts_times.push(inst_time);
            // println!("Instantiated done in {}ns", inst_time);

            for i in 0..$max {
                let inner: Vec<usize> = (i..=(i + i % $n + 1)).collect();
                collection.push(inner.into_boxed_slice());
            }
            let push_time = timer.elapsed().as_nanos() - inst_time;
            push_times.push(push_time);
            // println!("Push done in {}ms {}us {}ns", push_time / 1000000, (push_time / 1000) % 1000, push_time % 1000);

            let mut lookup_key: Vec<usize> = (($max - 1)..=(($max - 1) + ($max - 1) % $n) + 1).collect();
            
            let found = collection.iter().find(|k| {
                slice_eq(k, lookup_key.as_slice())
            }).unwrap().to_vec();
            
            let iter_time = timer.elapsed().as_nanos() - push_time;
            iter_times.push(iter_time);
            // println!("Iterator done in {}ms {}us {}ns", iter_time / 1000000, (iter_time / 1000) % 1000, iter_time % 1000);

            collection.iter_mut().for_each(|v| {
                v[0] += 1;
            });
            
            let iter_mut_time = timer.elapsed().as_nanos() - iter_time;
            iter_mut_times.push(iter_mut_time);
            // println!("Iter mut done in {}ms {}us {}ns", iter_mut_time / 1000000, (iter_mut_time / 1000) % 1000, iter_mut_time % 1000);
            
            collection.insert(0, vec![0].into_boxed_slice());
            
            let insert_time = timer.elapsed().as_nanos() - iter_mut_time;
            ins_front_times.push(insert_time);
            // println!("Insert front done in {}ms {}us {}ns", insert_time / 1000000, (insert_time / 1000) % 1000, insert_time % 1000);
            
            collection.insert(collection.len(), vec![0].into_boxed_slice());
            
            let insert_back_time = timer.elapsed().as_nanos() - insert_time;
            ins_back_times.push(insert_back_time);
            // println!("Insert back done in {}ms {}us {}ns", insert_back_time / 1000000, (insert_back_time / 1000) % 1000, insert_back_time % 1000);
            
            collection.remove(0);
            
            let remove_time = timer.elapsed().as_nanos() - insert_back_time;
            rm_front_times.push(remove_time);
            // println!("Remove front done in {}ms {}us {}ns", remove_time / 1000000, (remove_time / 1000) % 1000, remove_time % 1000);
            
            collection.remove(collection.len() - 1);
            
            let remove_back_time = timer.elapsed().as_nanos() - remove_time;
            rm_back_times.push(remove_back_time);
            // println!("Remove back done in {}ms {}us {}ns", remove_back_time / 1000000, (remove_back_time / 1000) % 1000, remove_back_time % 1000);

            let mut cloned = collection.clone();
            let clone_time = timer.elapsed().as_nanos() - remove_back_time;
            clone_times.push(clone_time);
            // println!("Clone done in {}ms {}us {}ns", clone_time / 1000000, (clone_time / 1000) % 1000, clone_time % 1000);

            for _ in 0..$max {
                cloned.pop();
            }

            let pop_time = timer.elapsed().as_nanos() - clone_time;
            pop_times.push(pop_time);
            // println!("Pop done in {}ms {}us {}ns", pop_time / 1000000, (pop_time / 1000) % 1000, pop_time % 1000);

            let mut last = vec![];
            lookup_key[0] += 1;
            for entry in collection {
                if &*entry == lookup_key.as_slice() {
                    last = entry.into()
                }
            }

            let into_time = timer.elapsed().as_nanos() - pop_time;
            into_iter_times.push(into_time);
            // println!("IntoIter done in {}ms {}us {}ns", into_time / 1000000, (into_time / 1000) % 1000, into_time % 1000);
            
            temp.push(found);
            temp.push(last);
        }

        // calculate mean for all measurement
        let insts_mean = insts_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let push_mean = push_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let iter_mean = iter_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let iter_mut_mean = iter_mut_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let ins_front_mean = ins_front_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let ins_back_mean = ins_back_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let rm_front_mean = rm_front_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let rm_back_mean = rm_back_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let clone_mean = clone_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let pop_mean = pop_times.iter().fold(0, |a, t| a + t) / $s as u128;
        let into_iter_mean = into_iter_times.iter().fold(0, |a, t| a + t) / $s as u128;

        // calculate variance for all measurement
        let insts_var = insts_times.iter().fold(0, |a, t| a + (t - insts_mean).pow(2)) / $s as u128;
        let push_var = push_times.iter().fold(0, |a, t| a + (t - push_mean).pow(2)) / $s as u128;
        let iter_var = iter_times.iter().fold(0, |a, t| a + (t - iter_mean).pow(2)) / $s as u128;
        let iter_mut_var = iter_mut_times.iter().fold(0, |a, t| a + (t - iter_mut_mean).pow(2)) / $s as u128;
        let ins_front_var = ins_front_times.iter().fold(0, |a, t| a + (t - ins_front_mean).pow(2)) / $s as u128;
        let ins_back_var = ins_back_times.iter().fold(0, |a, t| a + (t - ins_back_mean).pow(2)) / $s as u128;
        let rm_front_var = rm_front_times.iter().fold(0, |a, t| a + (t - rm_front_mean).pow(2)) / $s as u128;
        let rm_back_var = rm_back_times.iter().fold(0, |a, t| a + (t - rm_back_mean).pow(2)) / $s as u128;
        let clone_var = clone_times.iter().fold(0, |a, t| a + (t - clone_mean).pow(2)) / $s as u128;
        let pop_var = pop_times.iter().fold(0, |a, t| a + (t - pop_mean).pow(2)) / $s as u128;
        let into_iter_var = into_iter_times.iter().fold(0, |a, t| a + (t - into_iter_mean).pow(2)) / $s as u128;

        println!();
        println!("Average instantiate time is {}ns with sd = {}", insts_mean, (insts_var as f64).powf(0.5));
        println!("Average push time is {}ms {}us {}ns with sd = {}", push_mean / 1_000_000, (push_mean / 1_000) % 1_000, push_mean % 1000, (push_var as f64).powf(0.5));
        println!("Average iter time is {}ms {}us {}ns with sd = {}", iter_mean / 1_000_000, (iter_mean / 1_000) % 1_000, iter_mean % 1000, (iter_var as f64).powf(0.5));
        println!("Average iter mut time is {}ms {}us {}ns with sd = {}", iter_mut_mean / 1_000_000, (iter_mut_mean / 1_000) % 1_000, iter_mut_mean % 1000, (iter_mut_var as f64).powf(0.5));
        println!("Average insert front time is {}ms {}us {}ns with sd = {}", ins_front_mean / 1_000_000, (ins_front_mean / 1_000) % 1_000, ins_front_mean % 1000, (ins_front_var as f64).powf(0.5));
        println!("Average insert back time is {}ms {}us {}ns with sd = {}", ins_back_mean / 1_000_000, (ins_back_mean / 1_000) % 1_000, ins_back_mean % 1000, (ins_back_var as f64).powf(0.5));
        println!("Average remove front time is {}ms {}us {}ns with sd = {}", rm_front_mean / 1_000_000, (rm_front_mean / 1_000) % 1_000, rm_front_mean % 1000, (rm_front_var as f64).powf(0.5));
        println!("Average remove back time is {}ms {}us {}ns with sd = {}", rm_back_mean / 1_000_000, (rm_back_mean / 1_000) % 1_000, rm_back_mean % 1000, (rm_back_var as f64).powf(0.5));
        println!("Average clone time is {}ms {}us {}ns with sd = {}", clone_mean / 1_000_000, (clone_mean / 1_000) % 1_000, clone_mean % 1000, (clone_var as f64).powf(0.5));
        println!("Average pop time is {}ms {}us {}ns with sd = {}", pop_mean / 1_000_000, (pop_mean / 1_000) % 1_000, pop_mean % 1000, (pop_var as f64).powf(0.5));
        println!("Average into iter time is {}ms {}us {}ns with sd = {}", into_iter_mean / 1_000_000, (into_iter_mean / 1_000) % 1_000, into_iter_mean % 1000, (into_iter_var as f64).powf(0.5));
        temp.into_iter().fold(0, |a, i| a + i[0]);
    };
}

fn main() {
    const PRIME_MAX: usize = 1_000_000;
    const SAMPLE: usize = 1_000;
    bench_kv!(TupleVec, PRIME_MAX, SAMPLE);
    println!("================================");
    bench_kv!(VecPair, PRIME_MAX, SAMPLE);
    const NESTED_MAX: usize = 1_000;
    const INNER_MAX: usize = 1_000;
    println!("================================");
    bench_nested!(FlattenVec, NESTED_MAX, INNER_MAX, SAMPLE);
    println!("================================");
    bench_nested!(NestedVec, NESTED_MAX, INNER_MAX, SAMPLE);
}
