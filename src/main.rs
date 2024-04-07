mod wgpu {
    use burn::{
        backend::{
            wgpu::{Wgpu, WgpuDevice},
            Autodiff,
        },
        optim::{momentum::MomentumConfig, SgdConfig},
    };
    use burn_image_training::training::{train, TrainingConfig};
    use burn::backend::wgpu::AutoGraphicsApi;
     

    use burn_image_training::inference::infer;
    
    pub const MODEL_DIR: &str = "model";
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;

    pub fn run() {
        train::<Autodiff<Wgpu>>(
            TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
                momentum: 0.9,
                dampening: 0.,
                nesterov: false,
            }))),
            WgpuDevice::default(),
        );
    }

    pub fn inference() {
        infer::<MyBackend>(MODEL_DIR, WgpuDevice::default());
    }
}


fn main() {
    wgpu::inference();
}
