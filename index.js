import { DataLoader } from "./DataLoader.js";

async function main() {
    const data = new DataLoader();
    await data.load();
    await showExamples(data);
}

async function showExamples() {
    tfvis.visor().surface({ name: "My Surface", tab: "My Tab" });
}

main();
