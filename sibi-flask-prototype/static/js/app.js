// =============================
// Mobile nav: buka/tutup sidebar drawer
// =============================
const hamburgerBtn = document.getElementById("hamburgerBtn");
const sidebarCloseBtn = document.getElementById("sidebarCloseBtn");
const sidebarBackdrop = document.getElementById("sidebarBackdrop");
const sidebarEl = document.getElementById("sidebar");

function openSidebar() {
    if (!sidebarEl || !sidebarBackdrop) return;
    sidebarEl.classList.add("is-open");
    sidebarBackdrop.classList.add("is-open");
    if (hamburgerBtn) hamburgerBtn.setAttribute("aria-expanded", "true");
    document.body.style.overflow = "hidden";
}

function closeSidebar() {
    if (!sidebarEl || !sidebarBackdrop) return;
    sidebarEl.classList.remove("is-open");
    sidebarBackdrop.classList.remove("is-open");
    if (hamburgerBtn) hamburgerBtn.setAttribute("aria-expanded", "false");
    document.body.style.overflow = "";
}

if (hamburgerBtn) {
    hamburgerBtn.addEventListener("click", openSidebar);
}

if (sidebarCloseBtn) {
    sidebarCloseBtn.addEventListener("click", closeSidebar);
}

if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener("click", closeSidebar);
}

window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSidebar();
});


// =============================
// Real-time translate
// =============================
let realtimeStream = null;
let realtimeRunning = false;
let realtimeRequesting = false;
let realtimeInterval = null;

const predictionBuffer = [];
const SMOOTHING_WINDOW = 4;

let transcriptText = "";
let currentStableLabel = null;
let stableCount = 0;
let lastCommittedLabel = null;
let lastCommitTime = 0;

// Fitur landmark (63 angka) dari prediksi terakhir — dipakai untuk fitur
// "simpan data" (konfirmasi user), bukan buat prediksi itu sendiri.
let lastFeatures = null;

// 1:1 dengan karakter di transcriptText: { char, features }.
// Spasi disimpan dengan features: null (tidak ikut disimpan ke training).
let committedSamples = [];

const MIN_STABLE_COUNT = 2;
const LETTER_COOLDOWN = 1200;

const startRealtimeBtn = document.getElementById("startRealtimeBtn");
const stopRealtimeBtn = document.getElementById("stopRealtimeBtn");
const realtimePreview = document.getElementById("realtimePreview");
const realtimePlaceholder = document.getElementById("realtimePlaceholder");
const liveTranslationText = document.getElementById("liveTranslationText");
const realtimeStatus = document.getElementById("realtimeStatus");
const cameraCaption = document.getElementById("cameraCaption");

const deleteLastBtn = document.getElementById("deleteLastBtn");
const spaceBtn = document.getElementById("spaceBtn");
const clearTextBtn = document.getElementById("clearTextBtn");

const _canvas = document.createElement("canvas");
const _ctx = _canvas.getContext("2d");

const CAPTION_MAX_CHARS = 24;

function renderTranscript(fallbackText = "Belum ada teks.") {
    const fullText = transcriptText || fallbackText;

    if (liveTranslationText) {
        liveTranslationText.textContent = fullText;
    }

    // Caption di atas video kamera — supaya hasil translate kelihatan tanpa
    // perlu scroll ke kartu "Hasil Translate" di bawah (penting banget di HP).
    // Cuma nampilin potongan terakhir biar tetap satu baris & gak meluber
    // keluar layar kalau kalimatnya udah panjang — teks lengkapnya tetap
    // ada di kotak "Hasil translate" di bawah (yang bisa wrap ke bawah).
    if (cameraCaption) {
        const captionText =
            fullText.length > CAPTION_MAX_CHARS
                ? `…${fullText.slice(-CAPTION_MAX_CHARS)}`
                : fullText;

        cameraCaption.textContent = captionText;
        cameraCaption.classList.toggle("is-visible", realtimeRunning);
    }
}

function getSmoothedLabel(label) {
    if (!label) return null;

    // kalau label baru beda dari label stabil sebelumnya,
    // kosongkan buffer supaya tidak ketahan huruf lama
    if (currentStableLabel && label !== currentStableLabel) {
        predictionBuffer.length = 0;
    }

    predictionBuffer.push(label);

    if (predictionBuffer.length > SMOOTHING_WINDOW) {
        predictionBuffer.shift();
    }

    const counts = {};

    predictionBuffer.forEach((item) => {
        counts[item] = (counts[item] || 0) + 1;
    });

    return Object.keys(counts).reduce((a, b) =>
        counts[a] >= counts[b] ? a : b
    );
}

function updateTranscript(label) {
    if (!label) return;

    const now = Date.now();

    if (label === currentStableLabel) {
        stableCount++;
    } else {
        currentStableLabel = label;
        stableCount = 1;
    }

    if (
        stableCount >= MIN_STABLE_COUNT &&
        now - lastCommitTime > LETTER_COOLDOWN
    ) {
        transcriptText += label;
        lastCommittedLabel = label;
        lastCommitTime = now;
        stableCount = 0;

        committedSamples.push({ char: label, features: lastFeatures });
        renderFeedbackChips();
    }
}

function resetTranscript() {
    transcriptText = "";
    currentStableLabel = null;
    stableCount = 0;
    lastCommittedLabel = null;
    lastCommitTime = 0;
    predictionBuffer.length = 0;
    committedSamples = [];
    renderFeedbackChips();
}

function deleteLastCharacter() {
    if (!transcriptText) {
        renderTranscript("Belum ada huruf yang bisa dihapus.");
        return;
    }

    transcriptText = transcriptText.slice(0, -1);
    committedSamples.pop();

    currentStableLabel = null;
    stableCount = 0;
    lastCommittedLabel = null;
    lastCommitTime = Date.now();
    predictionBuffer.length = 0;

    renderTranscript("Huruf terakhir berhasil dihapus.");
    renderFeedbackChips();
}

function addSpace() {
    if (!transcriptText) {
        renderTranscript("Belum ada teks untuk diberi spasi.");
        return;
    }

    if (!transcriptText.endsWith(" ")) {
        transcriptText += " ";
        committedSamples.push({ char: " ", features: null });
    }

    currentStableLabel = null;
    stableCount = 0;
    lastCommittedLabel = null;
    lastCommitTime = Date.now();
    predictionBuffer.length = 0;

    renderTranscript();
    renderFeedbackChips();
}

function clearTranscript() {
    resetTranscript();
    renderTranscript("Teks berhasil direset.");
}

async function captureAndPredict() {
    if (!realtimeRunning || realtimeRequesting) return;
    if (!realtimePreview || realtimePreview.readyState < 2) return;

    realtimeRequesting = true;

    try {
        _canvas.width = realtimePreview.videoWidth || 640;
        _canvas.height = realtimePreview.videoHeight || 480;

        _ctx.drawImage(realtimePreview, 0, 0, _canvas.width, _canvas.height);

        const base64 = _canvas.toDataURL("image/jpeg", 0.7);

        const res = await fetch("/api/predict-frame", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ image: base64 }),
        });

        const data = await res.json();

        if (!realtimeRunning || !liveTranslationText) return;

        if (data.features) {
            lastFeatures = data.features;
        }

        if (data.label) {
            const smoothedLabel = getSmoothedLabel(data.label);

            updateTranscript(smoothedLabel);

            renderTranscript(smoothedLabel);
        } else {
            currentStableLabel = null;
            stableCount = 0;

            if (!transcriptText) {
                renderTranscript(data.message || "Tangan tidak terdeteksi");
            }
        }
    } catch (err) {
        console.error("predict-frame error:", err);

        if (liveTranslationText && realtimeRunning) {
            liveTranslationText.textContent = "Gagal memproses frame.";
        }
    } finally {
        realtimeRequesting = false;
    }
}

async function startRealtimeTranslate() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Browser kamu belum mendukung akses kamera.");
        return;
    }

    try {
        // Kalau layar lagi tegak (kayak HP dipegang normal), minta stream
        // kamera beraspek potret juga, biar nggak di-crop kasar pas
        // dipaksa masuk ke kotak tegak (sebelumnya selalu minta 640x480
        // landscape walau layarnya sempit & tinggi).
        const isPortrait = window.innerWidth < window.innerHeight;

        realtimeStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: isPortrait ? 480 : 640 },
                height: { ideal: isPortrait ? 640 : 480 },
                facingMode: "user",
            },
            audio: false,
        });

        realtimePreview.srcObject = realtimeStream;
        realtimePreview.style.display = "block";

        if (realtimePlaceholder) {
            realtimePlaceholder.style.display = "none";
        }

        startRealtimeBtn.disabled = true;
        stopRealtimeBtn.disabled = false;

        realtimeRunning = true;
        realtimeRequesting = false;
        resetTranscript();

        if (realtimeStatus) {
            realtimeStatus.innerHTML = `<span></span> Kamera aktif`;
        }

        if (liveTranslationText) {
            liveTranslationText.textContent = "Menunggu isyarat tangan...";
        }

        realtimeInterval = setInterval(captureAndPredict, 500);
    } catch (error) {
        alert("Kamera tidak bisa diakses. Pastikan izin kamera sudah diberikan di browser.");
        console.error(error);
    }
}

function stopRealtimeTranslate() {
    realtimeRunning = false;
    realtimeRequesting = false;

    if (realtimeInterval) {
        clearInterval(realtimeInterval);
        realtimeInterval = null;
    }

    if (realtimeStream) {
        realtimeStream.getTracks().forEach((track) => track.stop());
        realtimeStream = null;
    }

    if (realtimePreview) {
        realtimePreview.srcObject = null;
        realtimePreview.style.display = "none";
    }

    if (realtimePlaceholder) {
        realtimePlaceholder.style.display = "grid";
    }

    if (liveTranslationText) {
        liveTranslationText.textContent =
            transcriptText || "Real-time translate dihentikan.";
    }

    if (cameraCaption) {
        cameraCaption.classList.remove("is-visible");
    }

    if (realtimeStatus) {
        realtimeStatus.innerHTML = `<span></span> Kamera belum aktif`;
    }

    if (startRealtimeBtn) {
        startRealtimeBtn.disabled = false;
    }

    if (stopRealtimeBtn) {
        stopRealtimeBtn.disabled = true;
    }
}

if (startRealtimeBtn) {
    startRealtimeBtn.addEventListener("click", startRealtimeTranslate);
}

if (stopRealtimeBtn) {
    stopRealtimeBtn.addEventListener("click", stopRealtimeTranslate);
}

if (deleteLastBtn) {
    deleteLastBtn.addEventListener("click", deleteLastCharacter);
}

if (spaceBtn) {
    spaceBtn.addEventListener("click", addSpace);
}

if (clearTextBtn) {
    clearTextBtn.addEventListener("click", clearTranscript);
}


// =============================
// Tinjau kalimat penuh & simpan sample untuk training offline
// =============================
const feedbackCard = document.getElementById("feedbackCard");
const feedbackChips = document.getElementById("feedbackChips");
const feedbackSaveBtn = document.getElementById("feedbackSaveBtn");
const feedbackStatus = document.getElementById("feedbackStatus");

function removeCommittedChar(index) {
    transcriptText = transcriptText.slice(0, index) + transcriptText.slice(index + 1);
    committedSamples.splice(index, 1);

    renderTranscript();
    renderFeedbackChips();
}

function renderFeedbackChips() {
    if (!feedbackCard || !feedbackChips) return;

    if (!transcriptText) {
        feedbackCard.style.display = "none";
        feedbackChips.innerHTML = "";
        return;
    }

    feedbackCard.style.display = "block";
    if (feedbackStatus) feedbackStatus.textContent = "";
    feedbackChips.innerHTML = "";

    transcriptText.split("").forEach((char, index) => {
        const chip = document.createElement("span");
        chip.className = "feedback-chip" + (char === " " ? " is-space" : "");

        const label = document.createElement("span");
        label.textContent = char === " " ? "␣" : char;
        chip.appendChild(label);

        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "feedback-chip-remove";
        removeBtn.setAttribute("aria-label", `Hapus huruf ${char}`);
        removeBtn.textContent = "×";
        removeBtn.addEventListener("click", () => removeCommittedChar(index));
        chip.appendChild(removeBtn);

        feedbackChips.appendChild(chip);
    });
}

async function saveReviewedSamples() {
    const samples = committedSamples.filter((s) => s.features);

    if (!samples.length) {
        if (feedbackStatus) feedbackStatus.textContent = "Tidak ada huruf untuk disimpan.";
        return;
    }

    if (feedbackSaveBtn) feedbackSaveBtn.disabled = true;
    if (feedbackStatus) feedbackStatus.textContent = "Menyimpan...";

    let savedCount = 0;

    for (const sample of samples) {
        try {
            const res = await fetch("/api/save-sample", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    features: sample.features,
                    label: sample.char,
                    predicted_label: sample.char,
                }),
            });

            if (res.ok) savedCount++;
        } catch (err) {
            console.error("save-sample error:", err);
        }
    }

    if (feedbackSaveBtn) feedbackSaveBtn.disabled = false;

    if (feedbackStatus) {
        feedbackStatus.textContent =
            savedCount === samples.length
                ? `Tersimpan ${savedCount} huruf, makasih!`
                : `Tersimpan ${savedCount}/${samples.length} huruf (ada yang gagal).`;
    }
}

if (feedbackSaveBtn) {
    feedbackSaveBtn.addEventListener("click", saveReviewedSamples);
}


// =============================
// Stop camera when user leaves page
// =============================
window.addEventListener("beforeunload", () => {
    if (realtimeStream) {
        realtimeStream.getTracks().forEach((track) => track.stop());
    }
});


// =============================
// Kamus Abjad: search + modal preview
// =============================
const kamusSearch = document.getElementById("kamusSearch");
const kamusGrid = document.getElementById("kamusGrid");
const kamusEmpty = document.getElementById("kamusEmpty");
const kamusCards = kamusGrid ? Array.from(kamusGrid.querySelectorAll(".kamus-card")) : [];

const kamusModal = document.getElementById("kamusModal");
const kamusModalBackdrop = document.getElementById("kamusModalBackdrop");
const kamusModalClose = document.getElementById("kamusModalClose");
const kamusModalImg = document.getElementById("kamusModalImg");
const kamusModalLetter = document.getElementById("kamusModalLetter");

// Pindahkan modal jadi anak langsung <body>. Kalau dibiarkan di dalam
// .main-content, "position: fixed" jadi salah acuan karena .main-content
// punya animasi transform (pageFadeIn) yang membuat containing block baru,
// sehingga modal muncul kepotong / tidak center layar.
if (kamusModal) {
    document.body.appendChild(kamusModal);
}

function openKamusModal(letter, src) {
    if (!kamusModal || !kamusModalImg || !kamusModalLetter) return;

    kamusModalImg.src = src;
    kamusModalImg.alt = `Isyarat huruf ${letter}`;
    kamusModalLetter.textContent = letter;
    kamusModal.classList.add("is-open");
    document.body.style.overflow = "hidden";
}

function closeKamusModal() {
    if (!kamusModal) return;

    kamusModal.classList.remove("is-open");
    document.body.style.overflow = "";
}

kamusCards.forEach((card) => {
    card.addEventListener("click", () => {
        openKamusModal(card.dataset.letter, card.dataset.src);
    });
});

if (kamusModalBackdrop) {
    kamusModalBackdrop.addEventListener("click", closeKamusModal);
}

if (kamusModalClose) {
    kamusModalClose.addEventListener("click", closeKamusModal);
}

window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeKamusModal();
});

if (kamusSearch && kamusCards.length) {
    kamusSearch.addEventListener("input", () => {
        const query = kamusSearch.value.trim().toUpperCase();
        let visibleCount = 0;

        kamusCards.forEach((card) => {
            const matches = !query || card.dataset.letter.startsWith(query);
            card.classList.toggle("is-hidden", !matches);
            if (matches) visibleCount++;
        });

        if (kamusEmpty) {
            kamusEmpty.classList.toggle("is-visible", visibleCount === 0);
        }
    });
}