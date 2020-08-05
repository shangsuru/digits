import { DataLoader } from "./DataLoader.js";

async function main() {
    const TRAIN_SIZE = 5000;
    const VALIDATION_SIZE = 1000;
    const TEST_SIZE = 1000;
    const BATCH_SIZE = 512;
    const EPOCHS = 10;

    const data = new DataLoader(TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE);
    await data.load();
    const [testImages, testLabels] = data.getTestData();
    const [trainImages, trainLabels] = data.getTrainingData();
    const [validationImages, validationLabels] = data.getValidationData();

    await showExamples(testImages, 40);
    const model = createModel();
    tfvis.show.modelSummary({ name: "Architecture", tab: "Model" }, model);
    await trainModel(model, trainImages, trainLabels, validationImages, validationLabels, BATCH_SIZE, EPOCHS);
}

async function showExamples(examples, numExamples) {
    const surface = tfvis.visor().surface({ name: "Examples", tab: "Dataset", styles: { height: "1000px" } });
    // Loop through every example and show its image
    for (let i = 0; i < numExamples; i++) {
        const image = examples.slice([i, 0], [1, examples.shape[1]]);
        const canvas = document.createElement("canvas");
        canvas.className = "examples";
        await tf.browser.toPixels(image.reshape([28, 28, 1]), canvas);
        surface.drawArea.appendChild(canvas);
    }
}

function createModel() {
    const model = tf.sequential();
    model.add(
        tf.layers.conv2d({
            inputShape: [28, 28, 1],
            activation: "relu",
            kernelSize: 3,
            kernelInitializer: "varianceScaling",
            filters: 32,
        })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], stride: [2, 2] }));
    model.add(tf.layers.conv2d({ activation: "relu", kernelSize: 3, filters: 64, kernelInitializer: "varianceScaling" }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], stride: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 64, activation: "relu", kernelInitializer: "varianceScaling" }));
    model.add(tf.layers.dense({ units: 10, activation: "softmax", kernelInitializer: "varianceScaling" }));

    model.compile({
        optimizer: "rmsprop",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });
    return model;
}

function trainModel(model, trainX, trainY, valX, valY, batchSize, epochs) {
    model.fit(trainX, trainY, {
        batchSize,
        epochs,
        shuffle: true,
        validationData: [valX, valY],
        callbacks: tfvis.show.fitCallbacks({ name: "Training", tab: "Model", styles: { height: "1000px" } }, [
            "loss",
            "val_loss",
            "acc",
            "val_acc",
        ]),
    });
}

main();
