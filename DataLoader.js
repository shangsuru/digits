const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_SIZE = 28;

const DATA_SIZE = 65000;
const TRAIN_SIZE = 55000;
const TEST_SIZE = 10000;

const NUM_CLASSES = 10;

const MNIST_IMAGES_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const MNIST_LABELS_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

export class DataLoader {
    async load() {
        // Make request for the MNIST image set
        const image = new Image();
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");

        const imageRequest = new Promise((resolve, reject) => {
            image.crossOrigin = "";
            image.onload = () => {
                image.width = image.naturalWidth;
                image.height = image.naturalHeight;

                const datasetBytesBuffer = new ArrayBuffer(DATA_SIZE * IMAGE_SIZE * 4);

                const chunkSize = 5000;
                canvas.width = image.width;
                canvas.height = chunkSize;

                for (let i = 0; i < DATA_SIZE / chunkSize; i++) {
                    const viewBuffer = new Float32Array(
                        datasetBytesBuffer,
                        i * IMAGE_SIZE * chunkSize * 4,
                        IMAGE_SIZE * chunkSize
                    );
                    context.drawImage(image, 0, i * chunkSize, image.width, chunkSize, 0, 0, image.width, chunkSize);

                    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
                    for (let j = 0; j < imageData.data.length / 4; j++) {
                        viewBuffer[j] = imageData.data[j * 4] / 255;
                    }
                }

                this.datasetImages = new Float32Array(datasetBytesBuffer);
                resolve();
            };
            image.src = MNIST_IMAGES_PATH;
        });
        const labelsRequest = fetch(MNIST_LABELS_PATH);
        const [images, labels] = await Promise.all([imageRequest, labelsRequest]);
        this.dataSetLabels = new Uint8Array(await labels.arrayBuffer());

        this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * TRAIN_SIZE);
        this.trainLabels = this.dataSetLabels.slice(0, NUM_CLASSES * TRAIN_SIZE);
        this.testImages = this.datasetImages.slice(IMAGE_SIZE * TRAIN_SIZE);
        this.testLabels = this.dataSetLabels.slice(NUM_CLASSES * TRAIN_SIZE);
    }

    getTrainingData() {
        let x = tf.tensor4d(this.trainImages, [this.trainImages.length / IMAGE_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1]);
        let y = tf.tensor2d(this.trainLabels, [this.trainLabels.length / NUM_CLASSES, NUM_CLASSES]);

        return [x, y];
    }

    getTestData() {
        let x = tf.tensor4d(this.testImages, [this.testImages.length / IMAGE_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1]);
        let y = tf.tensor2d(this.testLabels, [this.testLabels.length / NUM_CLASSES, NUM_CLASSES]);

        return [x, y];
    }
}
