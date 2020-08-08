export function initCanvas(canvas, clearButton, predictButton, model) {
    const context = canvas.getContext("2d");
    canvas.width = 140;
    canvas.height = 140;
    let mousePressed = false;
    let xPosition;
    let yPosition;

    function onMouseDown(e) {
        console.log("mouse down");
        xPosition = e.pageX - this.offsetLeft - document.getElementsByClassName("visor")[0].offsetLeft;
        yPosition = e.pageY - this.offsetTop;
        mousePressed = true;
    }

    function onMouseUpOrLeave(e) {
        mousePressed = false;
    }

    function onMouseMove(e) {
        if (mousePressed) {
            let xPositionOld = xPosition;
            let yPositionOld = yPosition;
            xPosition = e.pageX - this.offsetLeft - document.getElementsByClassName("visor")[0].offsetLeft;
            yPosition = e.pageY - this.offsetTop;

            context.beginPath();
            context.strokeStyle = "black";
            context.lineWidth = 4;
            context.moveTo(xPositionOld, yPositionOld);
            context.lineTo(xPosition, yPosition);
            context.stroke();
            context.closePath();
        }
    }

    function clearCanvas() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        prediction.textContent = "";
    }

    function cropImage(image, width) {
        image = image.slice([0, 0, 3]);
        let mask_x = tf.greater(image.sum(0), 0).reshape([-1]);
        let mask_y = tf.greater(image.sum(1), 0).reshape([-1]);
        let st = tf.stack([mask_x, mask_y]);
        let v1 = tf.topk(st);
        let v2 = tf.topk(st.reverse());

        let [x1, y1] = v1.indices.dataSync();
        let [y2, x2] = v2.indices.dataSync();
        y2 = width - y2 - 1;
        x2 = width - x2 - 1;
        let crop_width = x2 - x1;
        let crop_height = y2 - y1;

        if (crop_width > crop_height) {
            y1 -= (crop_width - crop_height) / 2;
            crop_height = crop_width;
        }
        if (crop_height > crop_width) {
            x1 -= (crop_height - crop_width) / 2;
            crop_width = crop_height;
        }

        image = image.slice([y1, x1], [crop_height, crop_width]);
        image = image.pad([
            [6, 6],
            [6, 6],
            [0, 0],
        ]);
        let resized = tf.image.resizeNearestNeighbor(image, [28, 28]);

        for (let i = 0; i < 28 * 28; i++) {
            resized[i] = 255 - resized[i];
        }
        return resized;
    }

    function predict() {
        let image = tf.browser.fromPixels(canvas, 4);
        let resized = cropImage(image, canvas.width);
        let x_data = tf.cast(resized.reshape([1, 28, 28, 1]), "float32");

        let y_pred = model.predict(x_data);
        let prediction = Array.from(y_pred.argMax(1).dataSync());

        document.getElementById("prediction").innerText = prediction;

        const barchartData = Array.from(y_pred.dataSync()).map((d, i) => {
            return { index: i, value: d };
        });
        tfvis.render.barchart(document.getElementById("prediction-graph"), barchartData, { width: 400, height: 140 });
    }

    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", onMouseUpOrLeave);
    canvas.addEventListener("mouseleave", onMouseUpOrLeave);
    canvas.addEventListener("mousemove", onMouseMove);

    clearButton.onclick = clearCanvas;
    predictButton.onclick = predict;
}
