document.getElementById("predictBtn").addEventListener("click", async function () {

    const fileInput = document.getElementById("fileInput");
    const urlInput = document.getElementById("urlInput");
    const loading = document.getElementById("loading");
    const resultBox = document.getElementById("resultBox");

    // Reset UI
    resultBox.classList.add("hidden");
    loading.classList.remove("hidden");

    const formData = new FormData();

    // Validation
    if (fileInput.files.length > 0) {
        formData.append("image", fileInput.files[0]);
    } 
    else if (urlInput.value.trim() !== "") {
        formData.append("image_url", urlInput.value.trim());
    } 
    else {
        alert("Please upload an image or paste an image URL.");
        loading.classList.add("hidden");
        return;
    }

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server error occurred");
        }

        const data = await response.json();

        // Update images
        document.getElementById("originalImage").src =
            data.image + "?" + new Date().getTime();

        document.getElementById("gradcamImage").src =
            data.gradcam + "?" + new Date().getTime();

        // Prediction
        document.getElementById("finalPrediction").innerText =
            "Predicted Disease: " + data.prediction;

        // Confidence bar animation
        document.getElementById("confidenceBar").style.width =
            (data.confidence * 100) + "%";

        document.getElementById("confidenceText").innerText =
            "Confidence: " + (data.confidence * 100).toFixed(2) + "%";

        // Top 3
        let top3HTML = "";
        data.top3.forEach(item => {
            top3HTML += `
                <p>
                    ${item[0]} : ${(item[1] * 100).toFixed(2)}%
                </p>
            `;
        });

        document.getElementById("top3Results").innerHTML = top3HTML;

        resultBox.classList.remove("hidden");
    } 
    catch (error) {
        alert("Prediction failed. Please try again.");
        console.error(error);
    } 
    finally {
        loading.classList.add("hidden");
    }
});