use rand::random;

pub const FP_SHIFT: u32 = 16;
// 1.0 in fixed-point representation binary 1 followed by 16 zeros
pub const FP_ONE: i32 = 1 << FP_SHIFT;

#[repr(C, align(64))]
pub struct BinarizedLayer {
    input_chunks: usize,

    forward_weights: Vec<u64>,

    shadow_weights: Vec<i32>,

    thresholds: Vec<i32>,
}

impl BinarizedLayer {
    pub fn new(input_chunks: usize, num_neurons: usize) -> Self {
        let total_chunks = input_chunks * num_neurons;

        let mut forward_weights = vec![0; total_chunks];
        let mut shadow_weights = vec![0; total_chunks * 64];

        let mut rng_seed: u64 = 0x87654321_12345678 ^ random::<u64>();

        for chunk_idx in 0..total_chunks {
            rng_seed ^= rng_seed << 13;
            rng_seed ^= rng_seed >> 7;
            rng_seed ^= rng_seed << 17;

            let random_mask = rng_seed;
            forward_weights[chunk_idx] = random_mask;

            for bit_idx in 0..64 {
                let bit = (random_mask >> bit_idx) & 1;
                let shadow_idx = chunk_idx * 64 + bit_idx;

                shadow_weights[shadow_idx] = if bit == 1 { FP_ONE } else { -FP_ONE };
            }
        }

        Self {
            input_chunks,
            forward_weights,
            shadow_weights,
            thresholds: vec![((input_chunks * 64) / 2) as i32 * FP_ONE; num_neurons],
        }
    }

    pub fn forward(&self, inputs: &[u64], outputs: &mut [u64], num_neurons: usize) {
        outputs.fill(0);

        for neuron_idx in 0..num_neurons {
            let mut total_set_bits = 0;
            let weight_offset = neuron_idx * self.input_chunks;

            for chunk_idx in 0..self.input_chunks {
                let w = self.forward_weights[weight_offset + chunk_idx];
                let x = inputs[chunk_idx];
                total_set_bits += (!(x ^ w)).count_ones();
            }

            let fp_set_bits = (total_set_bits as i32) << FP_SHIFT;
            let fired = (fp_set_bits > self.thresholds[neuron_idx]) as u64;

            let out_chunk = neuron_idx >> 6;
            let out_bit = neuron_idx & 63;
            outputs[out_chunk] |= fired << out_bit;
        }
    }

    pub fn backward_ste(
        &mut self,
        inputs: &[u64],
        output_errors: &[i32],
        num_neurons: usize,
        mut prev_layer_errors: Option<&mut [i32]>,
        learning_rate_fp: i32,
    ) {
        if let Some(ref mut errs) = prev_layer_errors {
            errs.fill(0);
        }

        for neuron_idx in 0..num_neurons {
            let error_fp = output_errors[neuron_idx];
            if error_fp == 0 {
                continue;
            }

            let weight_offset = neuron_idx * self.input_chunks;

            for chunk_idx in 0..self.input_chunks {
                let in_chunk = inputs[chunk_idx];
                let forward_mask = self.forward_weights[weight_offset + chunk_idx];
                let shadow_offset = (weight_offset + chunk_idx) * 64;

                let mut new_bitmask: u64 = 0;

                for bit_idx in 0..64 {
                    if let Some(ref mut prev_errs) = prev_layer_errors {
                        let weight_bit = (forward_mask >> bit_idx) & 1;
                        let weight_dir = (weight_bit as i32 * 2) - 1;

                        let propagated = error_fp.saturating_mul(weight_dir);
                        let target_idx = chunk_idx * 64 + bit_idx;
                        prev_errs[target_idx] = prev_errs[target_idx].saturating_add(propagated);
                    }

                    let in_bit = (in_chunk >> bit_idx) & 1;
                    let in_dir = (in_bit as i32 * 2) - 1;

                    let delta = ((error_fp as i64 * learning_rate_fp as i64) >> FP_SHIFT) as i32;
                    let directional_delta = delta.saturating_mul(in_dir);

                    let s_idx = shadow_offset + bit_idx;
                    let clamp_max = FP_ONE + (FP_ONE >> 1);
                    let clamp_min = -clamp_max;

                    self.shadow_weights[s_idx] = self.shadow_weights[s_idx]
                        .saturating_add(directional_delta)
                        .clamp(clamp_min, clamp_max);

                    let bit_val = (self.shadow_weights[s_idx] >> 31) + 1;

                    new_bitmask |= (bit_val as u64) << bit_idx;
                }

                self.forward_weights[weight_offset + chunk_idx] = new_bitmask;
            }
        }
    }
}

/*
use rand::random_range;
use std::alloc::{Layout, alloc};
use std::mem::size_of;
use std::usize;

/// This X union is a ideomatic way to specify elements per a layer.
#[derive(Clone, Copy)]
#[repr(C, packed)]
pub union Layer {
    pub x: usize,
}

#[derive(Clone, Copy)]
pub enum ActFn {
    Linear,
    ReLU,
    LReLU,
    Sigmoid,
    Swish,
    Softmax,
    GeLU,
}

pub enum InitFunc {
    Zero,
    Constant { n: f64 },
    Prng { min: f64, max: f64 },
    XavierUniform,
}

/// This is simply a struct used to construct the memory arena.
#[repr(align(64))]
#[derive(Copy, Clone)]
pub struct ModelBuilder<Layer, const LAYERS: usize, const LEARNING_RATE: usize> {
    pub memory_layout: [Layer; LAYERS],
    pub activation_layout: [ActFn; LAYERS],
}

#[repr(align(64))]
pub struct Arena<const LAYERS: usize, const LEARNING_RATE: usize> {
    pub ptr: *mut f64,
    pub total_elements: usize,
    pub packed_endpoints: [usize; LAYERS],
    pub packed_acts: [ActFn; LAYERS],
}

impl<const LAYERS: usize, const LEARNING_RATE: usize> ModelBuilder<Layer, LAYERS, LEARNING_RATE> {
    pub fn allocate(self) -> Arena<LAYERS, LEARNING_RATE> {
        let mut packed_acts: [ActFn; LAYERS] = [ActFn::Linear; LAYERS];
        let mut packed_endpoints: [usize; LAYERS] = [0; LAYERS];
        let mut total_elements: usize = 0;

        unsafe {
            total_elements += self.memory_layout[0].x;
            packed_endpoints[0] += self.memory_layout[0].x;
        }
        // Note to self I started at 1 and not 0, because the input layer,
        // does not compute, or have bias, hench the unsafe block above.
        for i in 1..LAYERS {
            unsafe {
                // two because layer and bias
                total_elements += self.memory_layout[i].x * 2;
                // the weights are num of nodes of this layer times the upper
                total_elements += self.memory_layout[i].x * self.memory_layout[i - 1].x;
                packed_endpoints[i] = total_elements;
                packed_acts[i] = self.activation_layout[i];
            }
        }

        total_elements = total_elements.next_power_of_two();

        let bytes = total_elements
            .checked_mul(size_of::<f64>())
            .expect("Allocation size overflow");
        let layout = Layout::from_size_align(bytes, 64).expect("Invalid alignment/size");

        let raw_ptr = unsafe {
            let p = alloc(layout);
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            p as *mut f64
        };

        Arena {
            ptr: raw_ptr,
            total_elements,
            packed_endpoints,
            packed_acts,
        }
    }
}

impl<const LAYERS: usize, const LEARNING_RATE: usize> Arena<LAYERS, LEARNING_RATE> {
    pub fn init(&mut self, init: InitFunc) {
        unsafe {
            let mut curr = self.ptr.add(self.total_elements);

            while curr > self.ptr {
                curr = curr.sub(1);

                match init {
                    InitFunc::Zero => {
                        curr.write(0.0);
                    }
                    InitFunc::Constant { n } => curr.write(n),
                    InitFunc::Prng { min, max } => {
                        curr.write(random_range(min..max));
                    }
                    InitFunc::XavierUniform => {
                        todo!("XavierUniform is init in not complet as of yet!");
                    }
                }
            }
        }
    }

    /// Merely computes the activation functions for each layer, passing
    /// down the weights. Output is not returned, as it can be
    /// taken from the model.
    pub fn forward_pass(&mut self) {
        /*
        for i in 0..self.total_elements {
            unsafe {
                print!("{}, ", self.ptr.read());
                self.ptr = self.ptr.add(1);
            }
        }*/
        let mut mat_ptr = self.ptr;
        for layer in 0..LAYERS - 1 {
            let start = self.packed_endpoints[layer];
            let end = self.packed_endpoints[layer + 1];
            match self.packed_acts[layer] {
                ActFn::Linear => unsafe {
                    for _ in start..end {
                        let mut activation = self.ptr.read();

                        // inner matrix
                        todo!();

                        mat_ptr = mat_ptr.add(1);
                    }
                },
                ActFn::ReLU => {}
                ActFn::LReLU => {}
                ActFn::Sigmoid => {}
                ActFn::Swish => {}
                ActFn::Softmax => {}
                ActFn::GeLU => {}
            }
        }
    }
}
*/
