/**
 * Silent Voice - Modern Web Dashboard Client
 */

let state = {
    mode: "letters",
    status: "",
    top3: "",
    text: "",
    selected_lang: "english",
    translations: { english: "", hindi: "", kannada: "" },
    auto_speak: true,
    is_speaking: false,
    fps: 30
};

const LANG_META = {
    english: {
        flag: "🇬🇧",
        name: "English",
        border: "border-sky-500/40",
        bg: "bg-sky-50/70 dark:bg-sky-950/20",
        textCol: "text-slate-900 dark:text-white",
        isIndic: false,
        activeBtn: "bg-sky-600 text-white shadow-sky-500/20"
    },
    hindi: {
        flag: "🇮🇳",
        name: "हिन्दी (Hindi)",
        border: "border-emerald-500/40",
        bg: "bg-emerald-50/70 dark:bg-emerald-950/20",
        textCol: "text-emerald-800 dark:text-emerald-300",
        isIndic: true,
        activeBtn: "bg-emerald-600 text-white shadow-emerald-500/20"
    },
    kannada: {
        flag: "🟡",
        name: "ಕನ್ನಡ (Kannada)",
        border: "border-amber-500/40",
        bg: "bg-amber-50/70 dark:bg-amber-950/20",
        textCol: "text-amber-800 dark:text-amber-300",
        isIndic: true,
        activeBtn: "bg-amber-600 text-white shadow-amber-500/20"
    }
};

// Theme Management (Dark / Light Theme)
function initTheme() {
    const saved = localStorage.getItem("silent_voice_theme") || "dark";
    if (saved === "dark") {
        document.documentElement.classList.add("dark");
    } else {
        document.documentElement.classList.remove("dark");
    }
    updateThemeToggleIcons();
}

function toggleTheme() {
    const isDark = document.documentElement.classList.toggle("dark");
    const newTheme = isDark ? "dark" : "light";
    localStorage.setItem("silent_voice_theme", newTheme);
    updateThemeToggleIcons();
    showToast(`Switched to ${newTheme} theme`, "info");
}

function updateThemeToggleIcons() {
    const isDark = document.documentElement.classList.contains("dark");
    const suns = document.querySelectorAll(".theme-icon-sun");
    const moons = document.querySelectorAll(".theme-icon-moon");

    suns.forEach(sun => {
        if (isDark) sun.classList.remove("hidden");
        else sun.classList.add("hidden");
    });

    moons.forEach(moon => {
        if (isDark) moon.classList.add("hidden");
        else moon.classList.remove("hidden");
    });
}

let previousText = "";
let historyLog = [];
let hearingRecognition = null;
let isHearingListening = false;
let autoSmoothTimer = null;

document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    loadVocabulary();
    setupEventListeners();
    startStatePolling();
    loadCameraDevices();
    loadUserTranscripts();
});

function setupEventListeners() {
    const btnLetters = document.getElementById("btn-mode-letters");
    const btnWords = document.getElementById("btn-mode-words");
    if (btnLetters) btnLetters.addEventListener("click", () => setMode("letters"));
    if (btnWords) btnWords.addEventListener("click", () => setMode("words"));

    const btnSpace = document.getElementById("btn-space");
    const btnBackspace = document.getElementById("btn-backspace");
    const btnClear = document.getElementById("btn-clear");
    const btnSpeakAll = document.getElementById("btn-speak-all");

    if (btnSpace) btnSpace.addEventListener("click", () => sendAction("space"));
    if (btnBackspace) btnBackspace.addEventListener("click", () => sendAction("backspace"));
    if (btnClear) btnClear.addEventListener("click", () => {
        sendAction("clear");
        const banner = document.getElementById("ai-smoothed-banner");
        if (banner) banner.classList.add("hidden");
    });
    if (btnSpeakAll) btnSpeakAll.addEventListener("click", () => speakActiveText());

    const toggleAutoSpeak = document.getElementById("toggle-auto-speak");
    if (toggleAutoSpeak) {
        toggleAutoSpeak.addEventListener("change", (e) => {
            setAutoSpeak(e.target.checked);
        });
    }

    const searchInput = document.getElementById("vocab-search");
    if (searchInput) {
        searchInput.addEventListener("input", (e) => filterVocabulary(e.target.value));
    }

    document.addEventListener("keydown", handleKeydown);
}

function handleKeydown(e) {
    if (e.target && e.target.tagName === "INPUT") return;

    const key = e.key.toLowerCase();
    if (key === "l") {
        setMode("letters");
    } else if (key === "w") {
        setMode("words");
    } else if (key === "tab") {
        e.preventDefault();
        setMode(state.mode === "letters" ? "words" : "letters");
    } else if (key === " ") {
        e.preventDefault();
        sendAction("space");
    } else if (key === "backspace") {
        e.preventDefault();
        sendAction("backspace");
    } else if (key === "c") {
        sendAction("clear");
        const banner = document.getElementById("ai-smoothed-banner");
        if (banner) banner.classList.add("hidden");
    } else if (key === "s") {
        speakActiveText();
    } else if (key === "1") {
        setLanguage("english");
    } else if (key === "2") {
        setLanguage("hindi");
    } else if (key === "3") {
        setLanguage("kannada");
    }
}

// Poll state from FastAPI server
function startStatePolling() {
    setInterval(async () => {
        try {
            const res = await fetch("/api/state");
            if (res.status === 401) {
                window.location.href = "/login";
                return;
            }
            if (!res.ok) return;
            const data = await res.json();
            updateUI(data);
        } catch (err) {
            console.debug("State poll error:", err);
        }
    }, 150);
}

// Update DOM elements based on state
function updateUI(newState) {
    // Track new sentences for history log
    if (newState.text && newState.text !== state.text && newState.text.endsWith(" ")) {
        const trimmed = newState.text.trim();
        if (trimmed && (!historyLog.length || historyLog[0].text !== trimmed)) {
            addToHistory(trimmed, newState.translations);

            // Auto-smooth if enabled
            const autoSmoothToggle = document.getElementById("toggle-auto-smooth");
            if (autoSmoothToggle && autoSmoothToggle.checked) {
                if (autoSmoothTimer) clearTimeout(autoSmoothTimer);
                autoSmoothTimer = setTimeout(() => {
                    smoothCurrentSentence();
                }, 600);
            }
        }
    }

    const prevLang = state.selected_lang;
    state = newState;
    if (!state.selected_lang) state.selected_lang = prevLang || "english";

    // Update Mode indicator & buttons
    const modeBadge = document.getElementById("current-mode-badge");
    const hudMode = document.getElementById("hud-mode");
    const btnLetters = document.getElementById("btn-mode-letters");
    const btnWords = document.getElementById("btn-mode-words");

    if (modeBadge) {
        modeBadge.innerText = state.mode.toUpperCase();
        modeBadge.className = `px-2.5 py-1 text-xs font-bold rounded-full transition-all ${
            state.mode === "letters"
                ? "bg-sky-500/15 text-sky-700 dark:bg-sky-500/20 dark:text-sky-300 border border-sky-500/30"
                : "bg-indigo-500/15 text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-300 border border-indigo-500/30"
        }`;
    }
    if (hudMode) hudMode.innerText = state.mode.toUpperCase();

    if (btnLetters && btnWords) {
        if (state.mode === "letters") {
            btnLetters.className = "flex-1 sm:flex-none py-2 px-4 text-xs font-bold rounded-lg bg-sky-600 text-white shadow-md shadow-sky-600/20 transition-all";
            btnWords.className = "flex-1 sm:flex-none py-2 px-4 text-xs font-bold rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-all";
        } else {
            btnLetters.className = "flex-1 sm:flex-none py-2 px-4 text-xs font-bold rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-all";
            btnWords.className = "flex-1 sm:flex-none py-2 px-4 text-xs font-bold rounded-lg bg-indigo-600 text-white shadow-md shadow-indigo-600/20 transition-all";
        }
    }

    // Detected status & confidence meter
    const statusText = document.getElementById("detected-status");
    const hudStatus = document.getElementById("hud-status");
    const top3El = document.getElementById("top-guesses");
    const confidenceBar = document.getElementById("confidence-bar");
    const confidenceLabel = document.getElementById("confidence-label");

    if (statusText) statusText.innerText = state.status || "Ready / Show a sign";
    if (hudStatus) hudStatus.innerText = state.status || "Show Sign";
    if (top3El) top3El.innerText = state.top3 ? `Guesses: ${state.top3}` : "Awaiting gesture...";

    // Parse confidence percentage if present
    const confMatch = (state.status || "").match(/\((\d+)%\)/);
    const confVal = confMatch ? parseInt(confMatch[1], 10) : 0;

    if (confidenceBar && confidenceLabel) {
        confidenceBar.style.width = `${confVal}%`;
        confidenceLabel.innerText = `${confVal}%`;

        if (confVal >= 75) {
            confidenceBar.className = "h-full bg-emerald-500 rounded-full transition-all duration-300";
        } else if (confVal >= 45) {
            confidenceBar.className = "h-full bg-amber-500 rounded-full transition-all duration-300";
        } else {
            confidenceBar.className = "h-full bg-sky-500 rounded-full transition-all duration-300";
        }
    }

    // Update Main Active Language Output Card
    renderActiveText();

    // Update Secondary Comparison Cards (if open)
    const enText = document.getElementById("text-english");
    const hiText = document.getElementById("text-hindi");
    const knText = document.getElementById("text-kannada");

    if (enText) enText.innerText = state.translations.english || "—";
    if (hiText) hiText.innerText = state.translations.hindi || "—";
    if (knText) knText.innerText = state.translations.kannada || "—";

    // Auto-speak checkbox sync
    const toggleAutoSpeak = document.getElementById("toggle-auto-speak");
    if (toggleAutoSpeak && toggleAutoSpeak.checked !== state.auto_speak) {
        toggleAutoSpeak.checked = state.auto_speak;
    }

    // Speaking wave animation
    const speakingWave = document.getElementById("speaking-indicator");
    if (speakingWave) {
        speakingWave.style.display = state.is_speaking ? "inline-flex" : "none";
    }
}

// Render the focused active language card text
function renderActiveText() {
    const textEl = document.getElementById("active-lang-text");
    if (!textEl) return;

    const currentLang = state.selected_lang || "english";
    const val = (state.translations && state.translations[currentLang]) || "";

    if (val && val.trim()) {
        textEl.innerText = val;
        const meta = LANG_META[currentLang] || LANG_META.english;
        textEl.className = `text-2xl font-bold min-h-[4rem] break-words flex items-center ${meta.textCol} ${meta.isIndic ? "font-indic" : ""}`;
    } else {
        textEl.innerHTML = '<span class="text-slate-400 dark:text-slate-500 font-normal text-sm">Show signs to generate text...</span>';
    }
}

// Switch Active Language (English, Hindi, Kannada)
async function setLanguage(lang) {
    lang = (lang || "english").toLowerCase();
    state.selected_lang = lang;

    // Update pill buttons
    ["english", "hindi", "kannada"].forEach(l => {
        const btn = document.getElementById(`lang-btn-${l}`);
        if (!btn) return;
        const meta = LANG_META[l];
        if (l === lang) {
            btn.className = `lang-pill px-3 py-1.5 text-xs font-bold rounded-lg ${meta.activeBtn} shadow-md transition-all flex items-center space-x-1.5`;
        } else {
            btn.className = "lang-pill px-3 py-1.5 text-xs font-bold rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-all flex items-center space-x-1.5";
        }
    });

    // Update Main Active Output Card style & badge
    const card = document.getElementById("active-lang-card");
    const flag = document.getElementById("active-lang-flag");
    const name = document.getElementById("active-lang-name");
    const meta = LANG_META[lang] || LANG_META.english;

    if (card) {
        card.className = `glass-card rounded-2xl p-4 border ${meta.border} ${meta.bg} transition-all`;
    }
    if (flag) flag.innerText = meta.flag;
    if (name) name.innerText = meta.name;

    renderActiveText();
    showToast(`Active language: ${meta.name}`, "info");

    // Sync with backend
    try {
        await fetch("/api/language", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lang })
        });
    } catch (e) {
        console.error("Failed to sync language:", e);
    }
}

// Speak the text in the currently active language
function speakActiveText() {
    speakText(state.selected_lang || "english");
}

// Copy the text of the currently active language
function copyActiveText() {
    const currentLang = state.selected_lang || "english";
    const text = (state.translations && state.translations[currentLang]) || state.text;
    if (!text || !text.trim()) return;

    const meta = LANG_META[currentLang] || LANG_META.english;
    navigator.clipboard.writeText(text).then(() => {
        showToast(`${meta.name} text copied to clipboard!`, "success");
    }).catch(err => {
        console.error(err);
    });
}

// Toggle collapsible secondary languages
function toggleOtherLangs() {
    const cont = document.getElementById("other-langs-container");
    const chevron = document.getElementById("chevron-other-langs");
    if (!cont) return;

    const isHidden = cont.classList.contains("hidden");
    if (isHidden) {
        cont.classList.remove("hidden");
        if (chevron) chevron.style.transform = "rotate(180deg)";
    } else {
        cont.classList.add("hidden");
        if (chevron) chevron.style.transform = "rotate(0deg)";
    }
}

// API Actions
async function setMode(mode) {
    try {
        await fetch("/api/mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode })
        });
    } catch (e) {
        console.error(e);
    }
}

async function sendAction(action) {
    try {
        await fetch("/api/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action })
        });
    } catch (e) {
        console.error(e);
    }
}

async function speakText(lang) {
    try {
        let textToSpeak = state.translations[lang] || state.text;
        if (!textToSpeak || !textToSpeak.trim()) return;

        const meta = LANG_META[lang] || LANG_META.english;
        showToast(`Speaking in ${meta.name}...`, "info");
        await fetch("/api/speak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: textToSpeak, lang })
        });
    } catch (e) {
        console.error(e);
    }
}

async function setAutoSpeak(enabled) {
    try {
        await fetch("/api/auto_speak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled })
        });
        showToast(`Auto-speak ${enabled ? "enabled" : "disabled"}`, "info");
    } catch (e) {
        console.error(e);
    }
}

// Clipboard Copy Helper for secondary cards
function copyText(elementId, label) {
    const el = document.getElementById(elementId);
    if (!el || !el.innerText || el.innerText === "—") return;

    navigator.clipboard.writeText(el.innerText).then(() => {
        showToast(`${label} copied to clipboard!`, "success");
    }).catch(err => {
        console.error(err);
    });
}

// Toast notification
function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    const bgColors = {
        success: "bg-emerald-600/90 border-emerald-400/30 text-white",
        info: "bg-blue-600/90 border-blue-400/30 text-white",
        warning: "bg-amber-600/90 border-amber-400/30 text-white"
    };

    toast.className = `px-4 py-2.5 rounded-xl border backdrop-blur-md shadow-2xl flex items-center space-x-2 text-sm font-medium transition-all duration-300 ${bgColors[type] || bgColors.info}`;
    toast.innerHTML = `<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 2200);
}

// Load Vocabulary Explorer
let allVocabWords = [];

async function loadVocabulary() {
    try {
        const res = await fetch("/api/vocabulary");
        if (!res.ok) return;
        const data = await res.json();
        allVocabWords = data.words || [];
        renderVocabulary(allVocabWords);
    } catch (e) {
        console.error("Failed to load vocabulary:", e);
    }
}

function renderVocabulary(words) {
    const grid = document.getElementById("vocab-grid");
    if (!grid) return;

    grid.innerHTML = "";
    if (!words.length) {
        grid.innerHTML = '<div class="col-span-full py-8 text-center text-slate-500">No matching words found</div>';
        return;
    }

    words.forEach(item => {
        const card = document.createElement("div");
        card.className = "glass-card p-3 rounded-2xl border border-slate-200 dark:border-white/5 hover:border-sky-500/40 dark:hover:border-sky-500/40 transition-all flex flex-col justify-between group bg-white/80 dark:bg-slate-900/60";
        card.innerHTML = `
            <div>
                <div class="flex items-center justify-between mb-1.5">
                    <span class="text-[10px] font-bold text-sky-600 dark:text-sky-400 uppercase tracking-wider">${item.category || "ISL"}</span>
                    <button onclick="previewAudio('${item.english}', 'english')" class="p-1 rounded-md text-slate-400 hover:text-sky-600 dark:hover:text-sky-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition" title="Listen English">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"/></svg>
                    </button>
                </div>
                <div class="text-sm font-bold text-slate-800 dark:text-white capitalize group-hover:text-sky-600 dark:group-hover:text-sky-300 transition-colors">${item.english}</div>
            </div>
            <div class="mt-2 pt-2 border-t border-slate-100 dark:border-white/5 space-y-1">
                <div class="flex justify-between items-center text-xs">
                    <span class="text-slate-400">हिन्दी:</span>
                    <span class="font-semibold text-emerald-700 dark:text-emerald-400 font-indic">${item.hindi || "—"}</span>
                </div>
                <div class="flex justify-between items-center text-xs">
                    <span class="text-slate-400">ಕನ್ನಡ:</span>
                    <span class="font-semibold text-amber-700 dark:text-amber-400 font-indic">${item.kannada || "—"}</span>
                </div>
            </div>
        `;
        grid.appendChild(card);
    });
}

function filterVocabulary(query) {
    const q = (query || "").trim().toLowerCase();
    if (!q) {
        renderVocabulary(allVocabWords);
        return;
    }
    const filtered = allVocabWords.filter(w =>
        w.english.toLowerCase().includes(q) ||
        (w.hindi && w.hindi.toLowerCase().includes(q)) ||
        (w.kannada && w.kannada.toLowerCase().includes(q)) ||
        (w.category && w.category.toLowerCase().includes(q))
    );
    renderVocabulary(filtered);
}

function filterByCategory(cat) {
    const buttons = document.querySelectorAll(".vocab-cat-btn");
    buttons.forEach(b => b.classList.remove("bg-blue-600", "text-white"));

    if (cat === "all") {
        renderVocabulary(allVocabWords);
    } else {
        const filtered = allVocabWords.filter(w => (w.category || "").toLowerCase() === cat.toLowerCase());
        renderVocabulary(filtered);
    }
}

async function previewAudio(word, lang) {
    try {
        await fetch("/api/speak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: word, lang })
        });
    } catch (e) {
        console.error(e);
    }
}

// Camera controls
async function loadCameraDevices() {
    try {
        const res = await fetch("/api/camera/devices");
        if (!res.ok) return;
        const data = await res.json();
        const select = document.getElementById("camera-select");
        if (!select) return;

        select.innerHTML = (data.devices || []).map(d => `
            <option value="${d.index}" ${d.index === data.current ? 'selected' : ''}>${d.name}</option>
        `).join('');
    } catch (e) {
        console.error("Failed to load camera devices:", e);
    }
}

async function handleCameraChange(index) {
    try {
        const idx = parseInt(index, 10);
        showToast(`Switching to Camera ${idx}...`, "info");
        const res = await fetch("/api/camera/switch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ index: idx })
        });
        const d = await res.json();
        if (d.status === "ok") {
            showToast(`Camera ${idx} activated`, "success");
        }
    } catch (e) {
        showToast("Error switching camera", "warning");
    }
}

// AI natural sentence auto-smoothing
async function smoothCurrentSentence() {
    const raw = (state.text || "").trim();
    if (!raw) {
        showToast("Please sign words first to smooth into a sentence", "info");
        return;
    }

    const btn = document.getElementById("btn-smooth");
    if (btn) btn.innerHTML = "<span>✨ Smoothing...</span>";

    try {
        const res = await fetch("/api/smooth", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: raw })
        });
        const d = await res.json();

        // Display Smoothed Banner
        const banner = document.getElementById("ai-smoothed-banner");
        const smoothedTextEl = document.getElementById("ai-smoothed-text");
        const sourceBadge = document.getElementById("ai-source-badge");

        if (banner && smoothedTextEl) {
            banner.classList.remove("hidden");
            smoothedTextEl.innerText = d.smoothed;
            if (sourceBadge) {
                sourceBadge.innerText = d.source === "gemini" ? "Gemini AI" : "Natural Grammar AI";
            }
        }

        // Update active card text & secondary translations
        if (d.translations) {
            state.translations = d.translations;
            renderActiveText();

            const enText = document.getElementById("text-english");
            const hiText = document.getElementById("text-hindi");
            const knText = document.getElementById("text-kannada");

            if (enText) enText.innerText = d.translations.english || "—";
            if (hiText) hiText.innerText = d.translations.hindi || "—";
            if (knText) knText.innerText = d.translations.kannada || "—";
        }

        // Save smoothed entry to database
        fetch("/api/transcripts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                english: d.smoothed,
                hindi: d.translations.hindi,
                kannada: d.translations.kannada,
                raw_text: raw,
                speaker: "Signer",
                is_smoothed: true
            })
        }).then(() => loadUserTranscripts()).catch(console.error);

        showToast("✨ Sentence smoothed with natural grammar", "success");
    } catch (e) {
        console.error(e);
        showToast("Could not smooth sentence", "warning");
    } finally {
        if (btn) btn.innerHTML = "<span>✨ AI Smooth</span>";
    }
}

// Two-way speech to sign guide
function setupHearingSpeech() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
        showToast("Web Speech API not supported in this browser. Please use Chrome/Edge.", "warning");
        return null;
    }
    const rec = new SpeechRec();
    rec.continuous = false;
    rec.interimResults = false;

    rec.onstart = () => {
        isHearingListening = true;
        const lbl = document.getElementById("hearing-mic-label");
        const btn = document.getElementById("btn-hearing-mic");
        if (lbl) lbl.innerText = "Listening...";
        if (btn) btn.classList.add("animate-pulse", "ring-2", "ring-emerald-400");
    };

    rec.onresult = (event) => {
        const text = event.results[0][0].transcript;
        const input = document.getElementById("hearing-text-input");
        if (input) input.value = text;
        submitHearingText();
    };

    rec.onend = () => {
        isHearingListening = false;
        const lbl = document.getElementById("hearing-mic-label");
        const btn = document.getElementById("btn-hearing-mic");
        if (lbl) lbl.innerText = "Click & Speak";
        if (btn) btn.classList.remove("animate-pulse", "ring-2", "ring-emerald-400");
    };

    rec.onerror = () => {
        isHearingListening = false;
        const lbl = document.getElementById("hearing-mic-label");
        const btn = document.getElementById("btn-hearing-mic");
        if (lbl) lbl.innerText = "Click & Speak";
        if (btn) btn.classList.remove("animate-pulse", "ring-2", "ring-emerald-400");
    };

    return rec;
}

function toggleHearingSpeech() {
    if (!hearingRecognition) {
        hearingRecognition = setupHearingSpeech();
    }
    if (!hearingRecognition) return;

    if (isHearingListening) {
        hearingRecognition.stop();
    } else {
        const langSelect = document.getElementById("speech-lang-select");
        hearingRecognition.lang = langSelect ? langSelect.value : "en-IN";
        try {
            hearingRecognition.start();
        } catch (e) {
            console.error("Speech rec start error:", e);
        }
    }
}

async function submitHearingText() {
    const input = document.getElementById("hearing-text-input");
    if (!input || !input.value.trim()) return;

    const text = input.value.trim();
    const langSelect = document.getElementById("speech-lang-select");
    const lang = langSelect ? langSelect.value.split("-")[0] : "english";

    const container = document.getElementById("gesture-guide-container");
    const cards = document.getElementById("gesture-guide-cards");
    const countEl = document.getElementById("gesture-guide-count");

    if (container) container.classList.remove("hidden");
    if (cards) cards.innerHTML = '<div class="text-xs text-slate-400 py-2">Searching ISL gesture clips...</div>';

    try {
        const res = await fetch("/api/hearing/speak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, lang, auto_voice: false })
        });
        const data = await res.json();

        if (cards && data.references) {
            cards.innerHTML = "";
            let videoCount = 0;

            data.references.forEach(ref => {
                const card = document.createElement("div");
                card.className = "flex-shrink-0 w-32 p-2 rounded-xl bg-slate-900 border border-white/10 hover:border-emerald-500/40 transition flex flex-col items-center justify-between text-center space-y-1.5 shadow-md";

                if (ref.video_url) {
                    videoCount++;
                    card.innerHTML = `
                        <div class="relative w-full aspect-square rounded-lg overflow-hidden bg-black border border-white/10">
                            <video src="${ref.video_url}" autoplay loop muted playsinline class="w-full h-full object-cover"></video>
                            <span class="absolute top-1 right-1 px-1 py-0.5 rounded text-[8px] font-bold bg-emerald-500/80 text-black">ISL</span>
                        </div>
                        <div class="font-bold text-xs text-white capitalize truncate w-full">${ref.word}</div>
                        <span class="text-[9px] text-emerald-400 uppercase tracking-wider font-semibold">Sign Video</span>
                    `;
                } else {
                    const lettersStr = (ref.letters || []).join(" - ");
                    card.innerHTML = `
                        <div class="w-full aspect-square rounded-lg bg-blue-950/40 border border-blue-500/20 flex flex-col items-center justify-center p-1">
                            <span class="text-xs font-mono font-black text-blue-300 break-all">${lettersStr || ref.word}</span>
                        </div>
                        <div class="font-bold text-xs text-slate-200 capitalize truncate w-full">${ref.word}</div>
                        <span class="text-[9px] text-blue-400 uppercase tracking-wider font-semibold">Fingerspelling</span>
                    `;
                }
                cards.appendChild(card);
            });

            if (countEl) countEl.innerText = `${videoCount} video signs, ${data.references.length - videoCount} fingerspelled`;
        }

        showToast("Hearing message transcribed & sign guide displayed", "success");
        loadUserTranscripts();
    } catch (e) {
        console.error(e);
        showToast("Error processing hearing speech", "warning");
    }
}

// Session transcript history
async function loadUserTranscripts() {
    try {
        const res = await fetch("/api/transcripts");
        if (!res.ok) return;
        const d = await res.json();
        renderTranscriptsList(d.transcripts || []);
    } catch (e) {
        console.error("Failed to load transcripts:", e);
    }
}

function renderTranscriptsList(transcripts) {
    const list = document.getElementById("history-list");
    if (!list) return;

    if (!transcripts.length) {
        list.innerHTML = '<div class="text-xs text-slate-400 dark:text-slate-500 py-6 text-center">No session history logged yet. Sign gestures or speak into the microphone to log sentences.</div>';
        return;
    }

    list.innerHTML = "";
    transcripts.forEach((item, idx) => {
        const row = document.createElement("div");
        const isHearing = (item.speaker || "").toLowerCase().includes("hearing");
        const speakerBadgeClass = isHearing
            ? "bg-emerald-500/10 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400 border border-emerald-500/30"
            : "bg-sky-500/10 text-sky-700 dark:bg-sky-500/20 dark:text-sky-400 border border-sky-500/30";

        row.className = "p-3.5 rounded-2xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-white/5 space-y-1.5 text-xs shadow-sm hover:border-sky-500/30 transition";
        row.innerHTML = `
            <div class="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
                <div class="flex items-center space-x-2">
                    <span class="px-2 py-0.5 rounded-md font-bold uppercase tracking-wider ${speakerBadgeClass}">${item.speaker || "Signer"}</span>
                    ${item.is_smoothed ? '<span class="px-1.5 py-0.5 rounded text-[9px] bg-indigo-500/10 text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-300 border border-indigo-500/30">✨ AI Polished</span>' : ''}
                </div>
                <span class="font-mono text-slate-400 dark:text-slate-500">${item.timestamp || ""}</span>
            </div>
            <div class="text-sm font-semibold text-slate-900 dark:text-white">${item.english || "—"}</div>
            <div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-slate-600 dark:text-slate-400 font-indic pt-0.5">
                ${item.hindi ? `<span>🇮🇳 हिन्दी: <strong class="text-emerald-700 dark:text-emerald-400 font-medium">${item.hindi}</strong></span>` : ''}
                ${item.kannada ? `<span>🟡 ಕನ್ನಡ: <strong class="text-amber-700 dark:text-amber-400 font-medium">${item.kannada}</strong></span>` : ''}
            </div>
        `;
        list.appendChild(row);
    });
}

function exportTranscripts(fmt) {
    showToast(`Generating ${fmt.toUpperCase()} session report...`, "info");
    window.location.href = `/api/transcripts/export?format=${fmt}`;
}

async function clearUserTranscripts() {
    if (!confirm("Are you sure you want to clear your saved session transcript history?")) return;
    try {
        const res = await fetch("/api/transcripts", { method: "DELETE" });
        if (res.ok) {
            showToast("Session transcripts cleared", "success");
            loadUserTranscripts();
        }
    } catch (e) {
        showToast("Error clearing transcripts", "warning");
    }
}

// Add newly recognized word/sentence to history & database
function addToHistory(text, translations) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // Local array update
    historyLog.unshift({ time: timeStr, text, translations });
    if (historyLog.length > 50) historyLog.pop();

    // Persist to database
    fetch("/api/transcripts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            english: text,
            hindi: translations ? translations.hindi : "",
            kannada: translations ? translations.kannada : "",
            speaker: "Signer",
            raw_text: text,
            is_smoothed: false
        })
    }).then(() => loadUserTranscripts()).catch(console.error);
}

