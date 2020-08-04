import { DataLoader } from "./DataLoader.js";

async function main() {
    const data = new DataLoader();
    await data.load();
    await showExamples(data);
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
        tf.browser.toPixels(image.reshape([28, 28, 1]), canvas);
        surface.drawArea.appendChild(canvas);
    }
}

main();
