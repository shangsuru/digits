import { DataLoader } from "./DataLoader.js";

async function main() {
    const data = new DataLoader();
    await data.load();
    await showExamples(data);
    const model = getModel();
    tfvis.show.modelSummary({ name: "Model", tab: "Architecture" }, model);
}

async function showExamples(data) {
    const numExamples = 20;
    const surface = tfvis.visor().surface({ name: "Examples", tab: "Dataset" });
    const [examples, labels] = data.getTestData(numExamples);

    // Loop through every example and show its image
    for (let i = 0; i < numExamples; i++) {
        const image = examples.slice([i, 0], [1, examples.shape[1]]);
        const canvas = document.createElement("canvas");
        canvas.className = "examples";
        await tf.browser.toPixels(image.reshape([28, 28, 1]), canvas);
        surface.drawArea.appendChild(canvas);
    }
}

function getModel() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], activation: "relu", kernelSize: 3, filters: 32 }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], stride: [2, 2] }));
    model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], activation: "relu", kernelSize: 3, filters: 64 }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], stride: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

    model.compile({
        optimizer: "rmsprop",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });

    return model;
}

main();
