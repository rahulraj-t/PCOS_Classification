const form = document.getElementById("pcos-form");
const input = document.getElementById("image");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const dropText = document.getElementById("drop-text");
const errorBox = document.getElementById("error");
const result = document.getElementById("result");
const btn = document.getElementById("submit-btn");

function setError(msg) { errorBox.textContent = msg || ""; }

function chooseFile(file) {
  setError("");
  result.hidden = true;
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setError("Choose an image file (PNG, JPG, BMP or WEBP).");
    return;
  }
  const dt = new DataTransfer();
  dt.items.add(file);
  input.files = dt.files;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropText.innerHTML = `<strong>${file.name.replace(/</g, "&lt;")}</strong><br><small>Click to choose a different image</small>`;
  btn.disabled = false;
}

input.addEventListener("change", () => chooseFile(input.files[0]));

["dragenter", "dragover"].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", e => chooseFile(e.dataTransfer.files[0]));

function showResult({ affected, probability }) {
  const pct = Math.round(probability * 100);
  result.className = "result " + (affected ? "positive" : "negative");
  result.innerHTML = `
    <h2>${affected ? "Affected" : "Not affected"}</h2>
    <p>${affected
      ? "The model found signs in this image that match PCOS. Please consult a gynecologist or endocrinologist."
      : "The model did not find signs of PCOS in this image. If you have symptoms, please still talk to a doctor."}</p>
    <div class="meter" role="img" aria-label="Probability of PCOS ${pct} percent"><span style="width:${pct}%"></span></div>
    <p>Model probability of PCOS: <strong>${pct}%</strong></p>`;
  result.hidden = false;
  result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!input.files.length) { setError("Choose an image first."); return; }
  setError("");
  result.hidden = true;
  btn.disabled = true;
  btn.textContent = "Classifying…";

  try {
    const res = await fetch("/predict", { method: "POST", body: new FormData(form) });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.message || "Something went wrong.");
    showResult(data);
  } catch (err) {
    setError(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Classify image";
  }
});


/* CHATBOT */

const chatbotToggle = document.getElementById("chatbot-toggle");
const chatbotWindow = document.getElementById("chatbot-window");
const chatbotClose = document.getElementById("chatbot-close");
const chatbotForm = document.getElementById("chatbot-form");
const chatbotInput = document.getElementById("chatbot-input");
const chatbotMessages = document.getElementById("chatbot-messages");

chatbotToggle.addEventListener("click", () => {
  chatbotWindow.classList.toggle("hidden");
});

chatbotClose.addEventListener("click", () => {
  chatbotWindow.classList.add("hidden");
});

function appendMessage(text, sender) {
  const div = document.createElement("div");
  div.className = sender === "user"
    ? "user-message"
    : "bot-message";

  div.textContent = text;
  chatbotMessages.appendChild(div);
  chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
}

chatbotForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const message = chatbotInput.value.trim();

  if (!message) return;

  appendMessage(message, "user");
  chatbotInput.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: message
      })
    });

    const data = await response.json();

    appendMessage(data.reply, "bot");

  } catch (error) {
    appendMessage(
      "Sorry, I couldn't connect to the chatbot service.",
      "bot"
    );
  }
});