use NueralBinarizedRS::*;

fn main() {
    let mut layer0 = BinarizedLayer::new(2, 128);
    let mut layer1 = BinarizedLayer::new(1, 64);

    let input0: u64 = 7;
    let input1: u64 = 4;
    let target0: u64 = 3;
    let input_data = vec![input0, input1];

    let mut hidden_activations = vec![0; 1];
    let mut final_activations = vec![0; 1];

    let mut hidden_errors = vec![0; 64];
    let mut final_errors = vec![0; 64];

    let learning_rate_fp = (0.1 * (FP_ONE as f32)) as i32;

    println!("Target: {:064b}", target0);

    for epoch in 0..=2000 {
        layer0.forward(&input_data, &mut hidden_activations, 128);
        layer1.forward(&hidden_activations, &mut final_activations, 64);

        let current_output = final_activations[0];
        let error_bits = (current_output ^ target0).count_ones();

        if epoch % 200 == 0 || error_bits == 0 {
            println!(
                "Epoch {:4} | Bit Errors: {:2} | Out: {:064b}",
                epoch, error_bits, current_output
            );
            if error_bits == 0 {
                break;
            }
        }

        for i in 0..64 {
            let target_bit = (target0 >> i) & 1;
            let actual_bit = (current_output >> i) & 1;

            final_errors[i] = (target_bit as i32 - actual_bit as i32) * FP_ONE;
        }

        layer1.backward_ste(
            &hidden_activations,
            &final_errors,
            128,
            Some(&mut hidden_errors),
            learning_rate_fp,
        );

        layer0.backward_ste(&input_data, &hidden_errors, 64, None, learning_rate_fp);
    }
}

/*
// Run with cargo run --example xor_test_run
use NueralArenaRS::*;

const LAYERS: usize = 3;

const INPUTS: usize = 4;
const HIDDEN: usize = 8;
const OUTPUT: usize = 1;

fn main() {
    let model: ModelBuilder<Layer, LAYERS> = ModelBuilder {
        learning_rate: 0.1,
        memory_layout: [
            Layer { x: INPUTS },
            Layer { x: HIDDEN },
            Layer { x: OUTPUT },
        ],
        activation_layout: [ActFn::Sigmoid, ActFn::Sigmoid, ActFn::Linear],
    };
    let mut arena_model = model.allocate();
    arena_model.init(InitFunc::Zero);
    arena_model.forward_pass();
}
*/
