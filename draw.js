export function initCanvas(canvas) {
    const context = canvas.getContext("2d");
    canvas.width = 210;
    canvas.height = 210;
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
            context.lineWidth = 5;
            context.lineJoin = "round";
            context.moveTo(xPositionOld, yPositionOld);
            context.lineTo(xPosition, yPosition);
            context.stroke();
            context.closePath();
        }
    }

    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", onMouseUpOrLeave);
    canvas.addEventListener("mouseleave", onMouseUpOrLeave);
    canvas.addEventListener("mousemove", onMouseMove);
}
