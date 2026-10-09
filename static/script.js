
document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("pcos-form");
    const imageInput = document.getElementById("image");
    const preview = document.getElementById("preview");
    const dropzone = document.getElementById("dropzone");
    const dropText = document.getElementById("drop-text");
    const error = document.getElementById("error");
    const submitBtn = document.getElementById("submit-btn");
    const resetBtn = document.getElementById("reset-btn");

    const resultEmpty = document.getElementById("result-empty");
    const resultLoading = document.getElementById("result-loading");
    const result = document.getElementById("result");

    let selectedFile = null;
    let previewUrl = null;
    let isPredicting = false;

    if (!form || !imageInput || !preview || !dropzone) {
        console.error(
            "WomenSafe: Required image upload elements were not found. Check index.html IDs."
        );
        return;
    }

    /*
     * Read the maximum upload size from the existing HTML.
     * Example: "PNG, JPG, BMP or WEBP, up to 10 MB"
     */
    function getMaxFileSizeMB() {
        const text = dropText?.querySelector("small")?.textContent || "";
        const match = text.match(/([\d.]+)\s*MB/i);

        return match ? Number(match[1]) : 10;
    }

    function showError(message) {
        if (error) {
            error.textContent = message;
        }
    }

    function clearError() {
        if (error) {
            error.textContent = "";
        }
    }

    function releasePreviewUrl() {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
    }

    function showEmptyResult() {
        if (resultEmpty) resultEmpty.hidden = false;
        if (resultLoading) resultLoading.hidden = true;

        if (result) {
            result.hidden = true;
            result.replaceChildren();
        }
    }

    function clearPreview() {
        releasePreviewUrl();

        preview.onload = null;
        preview.onerror = null;
        preview.removeAttribute("src");
        preview.hidden = true;

        if (dropText) {
            dropText.hidden = false;
        }
    }

    function resetUpload() {
        if (isPredicting) return;

        imageInput.value = "";
        selectedFile = null;

        clearPreview();
        clearError();
        showEmptyResult();

        submitBtn.disabled = true;

        if (resetBtn) {
            resetBtn.hidden = true;
        }
    }

    function displayFile(file) {
        if (isPredicting) return;

        clearError();

        if (!file) return;

        const allowedTypes = [
            "image/jpeg",
            "image/png",
            "image/bmp",
            "image/webp"
        ];

        if (!allowedTypes.includes(file.type)) {
            resetUpload();
            showError("Please select a PNG, JPG, BMP, or WEBP image.");
            return;
        }

        const maxMB = getMaxFileSizeMB();

        if (file.size > maxMB * 1024 * 1024) {
            resetUpload();
            showError(
                `The image is too large. Please select an image up to ${maxMB} MB.`
            );
            return;
        }

        clearPreview();
        showEmptyResult();

        selectedFile = file;
        previewUrl = URL.createObjectURL(file);

        preview.onload = function () {
            preview.hidden = false;

            if (dropText) {
                dropText.hidden = true;
            }

            submitBtn.disabled = false;

            if (resetBtn) {
                resetBtn.hidden = false;
            }

            console.log(
                "WomenSafe: Image preview loaded successfully:",
                file.name
            );
        };

        preview.onerror = function () {
            clearPreview();
            selectedFile = null;
            imageInput.value = "";
            submitBtn.disabled = true;

            if (resetBtn) resetBtn.hidden = true;

            showError(
                "Unable to display this image. Please choose another image."
            );
        };

        preview.src = previewUrl;
    }

    /* ---------- Select an image from the computer ---------- */

    imageInput.addEventListener("change", function () {
        const file = imageInput.files?.[0];

        if (file) {
            displayFile(file);
        } else if (!isPredicting) {
            resetUpload();
        }
    });

    /* ---------- Drag-and-drop ---------- */

    ["dragenter", "dragover"].forEach(function (eventName) {
        dropzone.addEventListener(eventName, function (event) {
            event.preventDefault();
            event.stopPropagation();

            dropzone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach(function (eventName) {
        dropzone.addEventListener(eventName, function (event) {
            event.preventDefault();
            event.stopPropagation();

            dropzone.classList.remove("drag-over");
        });
    });

    dropzone.addEventListener("drop", function (event) {
        if (isPredicting) return;

        const file = event.dataTransfer?.files?.[0];

        if (!file) return;

        try {
            const transfer = new DataTransfer();
            transfer.items.add(file);
            imageInput.files = transfer.files;
        } catch (err) {
            console.warn("Could not update file input:", err);
        }

        displayFile(file);
    });

    /* ---------- Clear selected image ---------- */

    if (resetBtn) {
        resetBtn.addEventListener("click", function () {
            resetUpload();
        });
    }

    /* ---------- Display prediction response ---------- */

    function displayPrediction(data) {
        if (resultLoading) {
            resultLoading.hidden = true;
        }

        if (resultEmpty) {
            resultEmpty.hidden = true;
        }

        if (!result) return;

        result.replaceChildren();

        // Match the JSON returned by app.py
        if (data.ok !== true) {
            throw new Error(
                data.message || "Prediction failed."
            );
        }

        const probability = Number(data.probability);

        if (
            typeof data.affected !== "boolean" ||
            !Number.isFinite(probability) ||
            probability < 0 ||
            probability > 1
        ) {
            throw new Error(
                "The server returned invalid prediction data."
            );
        }

        const heading = document.createElement("h3");
        heading.textContent = "Analysis Result";
        result.appendChild(heading);

        const predictionText = document.createElement("p");

        predictionText.textContent = data.affected
            ? "Screening result: PCOS-positive"
            : "Screening result: PCOS-negative";

        result.appendChild(predictionText);

        const probabilityText = document.createElement("p");

        probabilityText.textContent =
            "Model-estimated probability: " +
            (probability * 100).toFixed(2) + "%";

        result.appendChild(probabilityText);

        const disclaimer = document.createElement("p");
        disclaimer.className = "result-disclaimer";

        disclaimer.textContent =
            "This is an AI-assisted screening estimate, not a medical diagnosis. Please consult a qualified healthcare professional.";

        result.appendChild(disclaimer);

        result.hidden = false;
    }


    /* ---------- Submit image to Flask ---------- */

    form.addEventListener("submit", async function (event) {
        event.preventDefault();

        if (isPredicting) return;

        if (!selectedFile) {
            showError("Please select an image before analysis.");
            return;
        }

        clearError();

        isPredicting = true;
        submitBtn.disabled = true;
        submitBtn.textContent = "Analysing...";

        if (resetBtn) {
            resetBtn.disabled = true;
        }

        if (resultEmpty) resultEmpty.hidden = true;
        if (result) result.hidden = true;
        if (resultLoading) resultLoading.hidden = false;

        try {
            const formData = new FormData();

            // Must match request.files["image"] in Flask.
            formData.append("image", selectedFile);

            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const contentType =
                response.headers.get("content-type") || "";

            let data;

            if (contentType.includes("application/json")) {
                data = await response.json();
            } else {
                const bodyText = await response.text();

                throw new Error(
                    "The server did not return JSON. Check the Flask /predict route. " +
                    bodyText.slice(0, 180)
                );
            }

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    data.message ||
                    `Prediction failed (HTTP ${response.status}).`
                );
            }

            displayPrediction(data);

        } catch (err) {
            console.error("WomenSafe prediction error:", err);

            if (resultLoading) resultLoading.hidden = true;
            if (resultEmpty) resultEmpty.hidden = false;

            showError(
                err.message ||
                "Unable to analyse the image. Please try again."
            );

        } finally {
            isPredicting = false;

            submitBtn.disabled = !selectedFile;
            submitBtn.textContent = "Analyse image";

            if (resetBtn) {
                resetBtn.disabled = false;
            }
        }
    });

    console.log("WomenSafe image upload initialized.");
});


/* =========================================================
   PART 2: CHATBOT
========================================================= */

/* ---------- Chatbot Elements ---------- */

const chatToggle = document.getElementById("chatbot-toggle");
const chatWindow = document.getElementById("chatbot-window");
const chatClose = document.getElementById("chatbot-close");
const chatForm = document.getElementById("chatbot-form");
const chatInput = document.getElementById("chatbot-input");
const chatMessages = document.getElementById("chatbot-messages");
const chips = document.getElementById("chips");


/* ---------- Open / Close Chatbot ---------- */

function setChat(open) {
    if (!chatWindow || !chatToggle) return;

    chatWindow.classList.toggle("hidden", !open);
    chatToggle.setAttribute("aria-expanded", String(open));

    if (open && chatInput) {
        chatInput.focus({ preventScroll: true });
    } else if (!open) {
        chatToggle.focus({ preventScroll: true });
    }
}

if (chatToggle) {
    chatToggle.addEventListener("click", function () {
        const isClosed = chatWindow.classList.contains("hidden");
        setChat(isClosed);
    });
}

if (chatClose) {
    chatClose.addEventListener("click", function () {
        setChat(false);
    });
}

document.addEventListener("keydown", function (event) {
    if (
        event.key === "Escape" &&
        chatWindow &&
        !chatWindow.classList.contains("hidden")
    ) {
        setChat(false);
    }
});


/* ---------- Scroll Only Inside Chat Messages ---------- */

function scrollChatToBottom() {
    if (!chatMessages) return;

    requestAnimationFrame(function () {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}


/* ---------- Add Chat Message ---------- */

function addMessage(text, who) {
    if (!chatMessages) return null;

    const messageElement = document.createElement("div");

    messageElement.className =
        who === "user" ? "user-message" : "bot-message";

    if (who === "bot" && window.marked && window.DOMPurify) {
        messageElement.innerHTML = window.DOMPurify.sanitize(
            window.marked.parse(text)
        );
    } else {
        messageElement.textContent = text;
    }

    chatMessages.appendChild(messageElement);
    scrollChatToBottom();

    return messageElement;
}


/* ---------- Send Message to Flask Chatbot ---------- */

async function sendChat(text) {
    if (!text || !text.trim()) return;

    addMessage(text, "user");

    if (chips) {
        chips.hidden = true;
    }

    const typingElement = addMessage("Typing...", "bot");

    if (typingElement) {
        typingElement.classList.add("typing");
    }

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: text
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error ||
                data.message ||
                `Chat request failed (HTTP ${response.status}).`
            );
        }

        const reply =
            data.reply ||
            "Sorry, I could not answer that.";

        if (typingElement) {
            typingElement.classList.remove("typing");

            if (window.marked && window.DOMPurify) {
                typingElement.innerHTML =
                    window.DOMPurify.sanitize(
                        window.marked.parse(reply)
                    );
            } else {
                typingElement.textContent = reply;
            }
        }

    } catch (err) {
        console.error("Chat request failed:", err);

        if (typingElement) {
            typingElement.classList.remove("typing");
            typingElement.textContent =
                "I can't reach the server right now. Please try again in a moment.";
        }
    }

    scrollChatToBottom();
}


/* ---------- Chat Form Submission ---------- */

if (chatForm) {
    chatForm.addEventListener("submit", function (event) {
        event.preventDefault();

        if (!chatInput) return;

        const message = chatInput.value.trim();

        if (!message) return;

        chatInput.value = "";
        chatInput.focus({ preventScroll: true });

        sendChat(message);
    });
}


/* ---------- Suggested Prompt Chips ---------- */

if (chips) {
    chips.addEventListener("click", function (event) {
        const chip = event.target.closest("button[data-q]");

        if (!chip) return;

        const message = chip.dataset.q?.trim();

        if (!message) return;

        sendChat(message);
    });
}
