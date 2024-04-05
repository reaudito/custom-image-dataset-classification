use crate::training::NUM_CLASSES;
use crate::{data::ClassificationBatcher, dataset::CIFAR10Loader, model::Cnn};
use burn::data::dataloader::{batcher::Batcher, Dataset};
use burn::{
    data::dataset::vision::ImageFolderDataset,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");
    let model: Cnn<B> = Cnn::new(NUM_CLASSES.into(), &device).load_record(record);
    let dataset = ImageFolderDataset::cifar10_test();

    let item = dataset.get(0).unwrap();
    let annotation = item.clone().annotation;
    let batcher = ClassificationBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    println!("Predicted {} Expected {:?}", predicted, annotation);
}
