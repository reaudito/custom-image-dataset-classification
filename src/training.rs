use std::time::Instant;

use crate::{
    data::{ClassificationBatch, ClassificationBatcher},
    dataset::CIFAR10Loader,
    model::Cnn,
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::ImageFolderDataset},
    nn::loss::CrossEntropyLossConfig,
    optim::SgdConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};
use burn::data::dataloader::Dataset;


const NUM_CLASSES: u8 = 10;
const ARTIFACT_DIR: &str = "/tmp/custom-image-dataset";

impl<B: Backend> Cnn<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<ClassificationBatch<B>, ClassificationOutput<B>> for Cnn<B> {
    fn step(&self, batch: ClassificationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ClassificationBatch<B>, ClassificationOutput<B>> for Cnn<B> {
    fn step(&self, batch: ClassificationBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: SgdConfig,
    #[config(default = 30)]
    pub num_epochs: usize,
    #[config(default = 128)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.02)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(ARTIFACT_DIR).ok();
    config
        .save(format!("{ARTIFACT_DIR}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    // Dataloaders
    let batcher_train = ClassificationBatcher::<B>::new(device.clone());
    let batcher_valid = ClassificationBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::cifar10_train());

    // NOTE: we use the CIFAR-10 test set as validation for demonstration purposes
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::cifar10_test());

    // Learner config
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            Cnn::new(NUM_CLASSES.into(), &device),
            config.optimizer.init(),
            config.learning_rate,
        );

    // Training
    let now = Instant::now();
    let model_trained = learner.fit(dataloader_train, dataloader_test);
    let elapsed = now.elapsed().as_secs();
    println!("Training completed in {}m{}s", (elapsed / 60), elapsed % 60);

    model_trained
        .save_file(format!("{ARTIFACT_DIR}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}


// pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) { 
//     let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
//         .expect("Config should exist for the model");
//     let record = CompactRecorder::new()
//         .load(format!("{artifact_dir}/model").into(), &device)
//         .expect("Trained model should exist");
//     let dataset = ImageFolderDataset::cifar10_test();

//     let value = dataset.get(0).unwrap();
// }


