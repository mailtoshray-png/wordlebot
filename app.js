const WORD_LENGTH = 5;
const MAX_GUESSES = 6;
const LOCAL_ALLOWED = "data/allowed.txt";
const LOCAL_ANSWERS = "data/answers.txt";
const LOCAL_FREQ_MAP = "data/freq_map.json";
const MIN_ALLOWED_SIZE = 12000;
const MIN_ANSWER_SIZE = 2000;
const PATTERN_COUNT = 3 ** WORD_LENGTH;
const scratchCounts = new Uint8Array(26);
const scratchResult = new Uint8Array(WORD_LENGTH);

const state = {
  mode: "solver",
  allowedWords: [],
  answerWords: [],
  remainingAnswers: [],
  currentRow: 0,
  pendingRow: null,
  pendingGuess: "",
  secret: "",
  guesses: [],
  keyStates: {},
  allowedCodes: [],
  answerCodes: [],
  remainingCodes: [],
  allowedIndex: new Map(),
  answerIndex: new Map(),
  priors: null,
  currentWeights: null,
  currentEntropy: null,
  stepStats: Array(MAX_GUESSES).fill(null),
  listsReady: false
};

const boardEl = document.getElementById("board");
const guessInput = document.getElementById("guess-input");
const enterButton = document.getElementById("enter");
const applyButton = document.getElementById("apply");
const resetButton = document.getElementById("reset");
const toggleModeButton = document.getElementById("toggle-mode");
const remainingCountEl = document.getElementById("remaining-count");
const bestGuessesEl = document.getElementById("best-guesses");
const miniBarsEl = document.getElementById("mini-bars");
const keyboardEl = document.getElementById("keyboard");
const modeLabelEl = document.getElementById("mode-label");
const patternHintEl = document.getElementById("pattern-hint");
const randomAnswerButton = document.getElementById("random-answer");
const revealAnswerButton = document.getElementById("reveal-answer");
const secretDisplayEl = document.getElementById("secret-display");
const progressEl = document.getElementById("progress");
const progressBarEl = progressEl.querySelector(".progress__bar");

function sanitizeGuess(value) {
  return value.toLowerCase().replace(/[^a-z]/g, "").slice(0, WORD_LENGTH);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function buildFrequencyPriors(freqMap, nCommon = 3000, width = 10) {
  const words = Object.keys(freqMap);
  const freqs = words.map((word) => freqMap[word]);
  const indices = freqs.map((_, idx) => idx).sort((a, b) => freqs[a] - freqs[b]);
  const n = words.length;
  const c = width * (-0.5 + nCommon / n);
  const start = c - width / 2;
  const step = n > 1 ? width / (n - 1) : 0;
  const priors = new Map();
  for (let rank = 0; rank < n; rank += 1) {
    const word = words[indices[rank]];
    const x = start + step * rank;
    priors.set(word, sigmoid(x));
  }
  return priors;
}

function getWeights(words, priors) {
  const weights = new Float64Array(words.length);
  let total = 0;
  for (let i = 0; i < words.length; i += 1) {
    const word = words[i];
    const weight = priors?.get(word) ?? 1;
    weights[i] = weight;
    total += weight;
  }
  if (!total) {
    const uniform = 1 / (words.length || 1);
    weights.fill(uniform);
    return weights;
  }
  for (let i = 0; i < weights.length; i += 1) {
    weights[i] /= total;
  }
  return weights;
}

function entropyFromDistribution(dist) {
  let entropy = 0;
  for (let i = 0; i < dist.length; i += 1) {
    const p = dist[i];
    if (p > 0) {
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

function formatNumber(value) {
  return Number.isFinite(value) ? value.toLocaleString("en-US") : "—";
}

function formatBits(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "—";
}

function setControlsEnabled(enabled) {
  guessInput.disabled = !enabled;
  enterButton.disabled = !enabled;
  applyButton.disabled = !enabled || state.pendingRow === null;
  toggleModeButton.disabled = !enabled;
  resetButton.disabled = !enabled;
  randomAnswerButton.disabled = !enabled;
  revealAnswerButton.disabled = !enabled;
}

function computeGainForGuess(guessCodes, answerCodes, weights, baseEntropy) {
  const buckets = new Float64Array(PATTERN_COUNT);
  for (let i = 0; i < answerCodes.length; i += 1) {
    const code = patternCodeFromCodes(guessCodes, answerCodes[i]);
    buckets[code] += weights[i];
  }
  const H1 = entropyFromDistribution(buckets);
  return { gain: baseEntropy - H1, H1 };
}

function getCurrentWeightsAndEntropy() {
  const weights = getWeights(state.remainingAnswers, state.priors);
  const H0 = entropyFromDistribution(weights);
  return { weights, H0 };
}

function getActiveWeightsAndEntropy() {
  if (state.currentWeights && state.currentWeights.length === state.remainingAnswers.length) {
    const H0 =
      state.currentEntropy !== null && state.currentEntropy !== undefined
        ? state.currentEntropy
        : entropyFromDistribution(state.currentWeights);
    return { weights: state.currentWeights, H0 };
  }
  return getCurrentWeightsAndEntropy();
}

function entropyToExpectedScore(entropy) {
  const twoToMinus = 2 ** -entropy;
  const minScore = twoToMinus + 2 * (1 - twoToMinus);
  return minScore + (1.5 * entropy) / 11.5;
}

function wordToCodes(word) {
  const codes = new Uint8Array(WORD_LENGTH);
  for (let i = 0; i < WORD_LENGTH; i += 1) {
    codes[i] = word.charCodeAt(i) - 97;
  }
  return codes;
}

function patternCodeFromCodes(guessCodes, answerCodes) {
  scratchCounts.fill(0);
  scratchResult.fill(0);
  for (let i = 0; i < WORD_LENGTH; i += 1) {
    if (guessCodes[i] === answerCodes[i]) {
      scratchResult[i] = 2;
    } else {
      scratchCounts[answerCodes[i]] += 1;
    }
  }
  for (let i = 0; i < WORD_LENGTH; i += 1) {
    if (scratchResult[i] === 0) {
      const letter = guessCodes[i];
      if (scratchCounts[letter]) {
        scratchResult[i] = 1;
        scratchCounts[letter] -= 1;
      }
    }
  }
  let code = 0;
  let factor = 1;
  for (let i = 0; i < WORD_LENGTH; i += 1) {
    code += scratchResult[i] * factor;
    factor *= 3;
  }
  return code;
}

function getBoardRow(rowIndex) {
  return boardEl.querySelector(`.board-row[data-row="${rowIndex}"]`);
}

function buildBoard() {
  boardEl.innerHTML = "";
  for (let r = 0; r < MAX_GUESSES; r += 1) {
    const rowWrap = document.createElement("div");
    rowWrap.className = "board-row";
    rowWrap.dataset.row = r;

    const left = document.createElement("div");
    left.className = "row-left";
    const leftPos = document.createElement("span");
    leftPos.className = "row-left__pos";
    leftPos.textContent = "—";
    const leftBits = document.createElement("span");
    leftBits.className = "row-left__bits";
    leftBits.textContent = "—";
    left.appendChild(leftPos);
    left.appendChild(leftBits);

    const row = document.createElement("div");
    row.className = "row";
    row.dataset.row = r;

    for (let c = 0; c < WORD_LENGTH; c += 1) {
      const tile = document.createElement("div");
      tile.className = "tile";
      tile.dataset.col = c;
      const letter = document.createElement("span");
      letter.textContent = "";
      tile.appendChild(letter);
      row.appendChild(tile);
    }
    const right = document.createElement("div");
    right.className = "row-right";
    right.textContent = "—";

    rowWrap.appendChild(left);
    rowWrap.appendChild(row);
    rowWrap.appendChild(right);
    boardEl.appendChild(rowWrap);
  }
}

function buildKeyboard() {
  keyboardEl.innerHTML = "";
  const rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];
  rows.forEach((rowLetters, rowIndex) => {
    const row = document.createElement("div");
    row.className = "key-row";
    if (rowIndex === 2) {
      const enterKey = createKey("enter", "Enter", true);
      row.appendChild(enterKey);
    }
    rowLetters.split("").forEach((letter) => {
      row.appendChild(createKey(letter, letter.toUpperCase()));
    });
    if (rowIndex === 2) {
      const backKey = createKey("backspace", "Back", true);
      row.appendChild(backKey);
    }
    keyboardEl.appendChild(row);
  });
}

function createKey(code, label, wide = false) {
  const key = document.createElement("button");
  key.className = `key${wide ? " key--wide" : ""}`;
  key.dataset.key = code;
  key.textContent = label;
  key.type = "button";
  return key;
}

function updateRowLetters(rowIndex, word) {
  const rowWrap = getBoardRow(rowIndex);
  const row = rowWrap?.querySelector(".row");
  if (!row) return;
  [...row.children].forEach((tile, idx) => {
    const span = tile.querySelector("span");
    span.textContent = word[idx] ? word[idx].toUpperCase() : "";
    if (!state.guesses[rowIndex] && state.pendingRow !== rowIndex) {
      clearTileState(tile);
    }
  });
}

function clearTileState(tile) {
  tile.classList.remove("tile--gray", "tile--yellow", "tile--green", "tile--pending");
}

function setTileState(tile, stateName) {
  clearTileState(tile);
  if (stateName) {
    tile.classList.add(`tile--${stateName}`);
  }
}

function applyPatternToRow(rowIndex, pattern, pending = false) {
  const rowWrap = getBoardRow(rowIndex);
  const row = rowWrap?.querySelector(".row");
  if (!row) return;
  [...row.children].forEach((tile, idx) => {
    const status = pattern[idx];
    if (status === 2) setTileState(tile, "green");
    else if (status === 1) setTileState(tile, "yellow");
    else setTileState(tile, "gray");
    if (pending) tile.classList.add("tile--pending");
  });
}

function updateRowStats(rowIndex, remaining, entropy, gain) {
  const rowWrap = getBoardRow(rowIndex);
  if (!rowWrap) return;
  const leftPos = rowWrap.querySelector(".row-left__pos");
  const leftBits = rowWrap.querySelector(".row-left__bits");
  const right = rowWrap.querySelector(".row-right");
  if (leftPos) leftPos.textContent = `${formatNumber(remaining)} Pos,`;
  if (leftBits) leftBits.textContent = `${formatBits(entropy)} Bits`;
  if (right) right.textContent = `${formatBits(gain)} Bits`;
}

function clearRowStats(rowIndex) {
  const rowWrap = getBoardRow(rowIndex);
  if (!rowWrap) return;
  const leftPos = rowWrap.querySelector(".row-left__pos");
  const leftBits = rowWrap.querySelector(".row-left__bits");
  const right = rowWrap.querySelector(".row-right");
  if (leftPos) leftPos.textContent = "—";
  if (leftBits) leftBits.textContent = "—";
  if (right) right.textContent = "—";
}

function clearAllRowStats() {
  for (let r = 0; r < MAX_GUESSES; r += 1) {
    clearRowStats(r);
  }
}

function patternToCode(pattern) {
  let code = 0;
  let factor = 1;
  for (let i = 0; i < pattern.length; i += 1) {
    code += pattern[i] * factor;
    factor *= 3;
  }
  return code;
}

function codeToPattern(code) {
  const pattern = Array(WORD_LENGTH).fill(0);
  let remaining = code;
  for (let i = 0; i < WORD_LENGTH; i += 1) {
    pattern[i] = remaining % 3;
    remaining = Math.floor(remaining / 3);
  }
  return pattern;
}

function patternFromRow(rowIndex) {
  const rowWrap = getBoardRow(rowIndex);
  const row = rowWrap?.querySelector(".row");
  if (!row) return Array(WORD_LENGTH).fill(0);
  return [...row.children].map((tile) => {
    if (tile.classList.contains("tile--green")) return 2;
    if (tile.classList.contains("tile--yellow")) return 1;
    return 0;
  });
}

function filterRemaining(guess, pattern) {
  const code = patternToCode(pattern);
  const guessIndex = state.allowedIndex.get(guess);
  const guessCodes = Number.isInteger(guessIndex)
    ? state.allowedCodes[guessIndex]
    : wordToCodes(guess);
  const nextAnswers = [];
  const nextCodes = [];
  for (let i = 0; i < state.remainingAnswers.length; i += 1) {
    const answerCodes = state.remainingCodes[i];
    const patternCode = patternCodeFromCodes(guessCodes, answerCodes);
    if (patternCode === code) {
      nextAnswers.push(state.remainingAnswers[i]);
      nextCodes.push(answerCodes);
    }
  }
  state.remainingAnswers = nextAnswers;
  state.remainingCodes = nextCodes;
}

function updateStats(bestScore = null) {
  if (remainingCountEl) {
    remainingCountEl.textContent = `${formatNumber(state.remainingAnswers.length)} Pos`;
  }
  if (modeLabelEl) {
    modeLabelEl.textContent = state.mode === "solver" ? "Solver" : "Play";
  }
}

function updateKeyboardFromPattern(guess, pattern) {
  for (let i = 0; i < WORD_LENGTH; i += 1) {
    const letter = guess[i];
    const status = pattern[i];
    const current = state.keyStates[letter] || 0;
    if (status > current) {
      state.keyStates[letter] = status;
    }
  }
  applyKeyboardStyles();
}

function applyKeyboardStyles() {
  keyboardEl.querySelectorAll(".key").forEach((key) => {
    const letter = key.dataset.key;
    if (!letter || letter.length !== 1) return;
    key.classList.remove("key--gray", "key--yellow", "key--green");
    const status = state.keyStates[letter];
    if (status === 2) key.classList.add("key--green");
    else if (status === 1) key.classList.add("key--yellow");
    else if (status === 0 && letter in state.keyStates) key.classList.add("key--gray");
  });
}

function setSecret(word = null) {
  if (!state.answerWords.length) {
    state.secret = "";
    if (secretDisplayEl) secretDisplayEl.textContent = "Secret: —";
    return;
  }
  const selection = word || state.answerWords[Math.floor(Math.random() * state.answerWords.length)];
  state.secret = selection;
  if (secretDisplayEl) secretDisplayEl.textContent = "Secret: —";
}

function revealSecret() {
  if (!state.secret) return;
  if (secretDisplayEl) {
    secretDisplayEl.textContent = `Secret: ${state.secret.toUpperCase()}`;
  }
}

function clearBoard() {
  buildBoard();
  state.currentRow = 0;
  state.pendingRow = null;
  state.pendingGuess = "";
  state.guesses = [];
  state.keyStates = {};
  state.stepStats = Array(MAX_GUESSES).fill(null);
  applyKeyboardStyles();
  updateRowLetters(0, "");
  clearAllRowStats();
  if (miniBarsEl) miniBarsEl.innerHTML = "";
  guessInput.value = "";
  guessInput.disabled = false;
  enterButton.disabled = false;
  applyButton.disabled = true;
  patternHintEl.textContent =
    state.mode === "solver"
      ? "Solver mode: enter a guess, then click tiles to set colors, then press Apply."
      : "Play mode: enter guesses to solve the hidden word.";
}

function resetGame() {
  state.remainingAnswers = [...state.answerWords];
  state.remainingCodes = state.answerCodes.slice();
  state.currentWeights = null;
  state.currentEntropy = null;
  clearBoard();
  if (state.mode === "play") {
    setSecret();
  }
  updateStats();
  if (state.listsReady) {
    computeSuggestions();
  }
}

function handleEnter() {
  if (state.pendingRow !== null) return;
  let guess = sanitizeGuess(guessInput.value);
  if (guess.length !== WORD_LENGTH) return;
  if (!state.allowedWords.includes(guess)) {
    patternHintEl.textContent = "Guess not in allowed list.";
    return;
  }

  updateRowLetters(state.currentRow, guess);

  if (state.mode === "play") {
    const guessIndex = state.allowedIndex.get(guess);
    const guessCodes = Number.isInteger(guessIndex)
      ? state.allowedCodes[guessIndex]
      : wordToCodes(guess);
    const { weights, H0 } = getActiveWeightsAndEntropy();
    const { gain } = computeGainForGuess(guessCodes, state.remainingCodes, weights, H0);
    updateRowStats(state.currentRow, state.remainingAnswers.length, H0, gain);
    state.stepStats[state.currentRow] = { remaining: state.remainingAnswers.length, entropy: H0, gain };
    const answerIndex = state.answerIndex.get(state.secret);
    const answerCodes = Number.isInteger(answerIndex)
      ? state.answerCodes[answerIndex]
      : wordToCodes(state.secret);
    const patternCode = patternCodeFromCodes(guessCodes, answerCodes);
    const pattern = codeToPattern(patternCode);
    applyPatternToRow(state.currentRow, pattern);
    state.guesses.push({ guess, pattern });
    updateKeyboardFromPattern(guess, pattern);
    filterRemaining(guess, pattern);
    state.currentRow += 1;
    guessInput.value = "";
    updateRowLetters(state.currentRow, "");

    if (guess === state.secret) {
      patternHintEl.textContent = "Solved!";
      revealSecret();
      return;
    }
    if (state.currentRow >= MAX_GUESSES) {
      patternHintEl.textContent = "Out of guesses.";
      revealSecret();
      return;
    }
    computeSuggestions();
  } else {
    state.pendingRow = state.currentRow;
    state.pendingGuess = guess;
    const pendingPattern = Array(WORD_LENGTH).fill(0);
    applyPatternToRow(state.pendingRow, pendingPattern, true);
    applyButton.disabled = false;
    guessInput.disabled = true;
    enterButton.disabled = true;
    patternHintEl.textContent = "Click tiles to set feedback, then press Apply.";
  }
}

function handleApply() {
  if (state.pendingRow === null) return;
  const pattern = patternFromRow(state.pendingRow);
  const guess = state.pendingGuess;

  const guessIndex = state.allowedIndex.get(guess);
  const guessCodes = Number.isInteger(guessIndex)
    ? state.allowedCodes[guessIndex]
    : wordToCodes(guess);
  const { weights, H0 } = getActiveWeightsAndEntropy();
  const { gain } = computeGainForGuess(guessCodes, state.remainingCodes, weights, H0);
  updateRowStats(state.pendingRow, state.remainingAnswers.length, H0, gain);
  state.stepStats[state.pendingRow] = { remaining: state.remainingAnswers.length, entropy: H0, gain };

  filterRemaining(guess, pattern);
  state.guesses.push({ guess, pattern });
  updateKeyboardFromPattern(guess, pattern);
  applyPatternToRow(state.pendingRow, pattern);

  state.pendingRow = null;
  state.pendingGuess = "";
  state.currentRow += 1;
  applyButton.disabled = true;
  guessInput.disabled = false;
  enterButton.disabled = false;
  patternHintEl.textContent = "Solver mode: enter a guess, then click tiles to set colors, then press Apply.";
  guessInput.value = "";
  updateRowLetters(state.currentRow, "");
  computeSuggestions();
}

function handleTileClick(event) {
  if (state.mode !== "solver" || state.pendingRow === null) return;
  const tile = event.target.closest(".tile");
  if (!tile) return;
  const row = tile.closest(".row");
  if (!row || Number(row.dataset.row) !== state.pendingRow) return;

  if (tile.classList.contains("tile--gray")) {
    setTileState(tile, "yellow");
  } else if (tile.classList.contains("tile--yellow")) {
    setTileState(tile, "green");
  } else {
    setTileState(tile, "gray");
  }
}

function computeSuggestions() {
  if (!state.listsReady) {
    bestGuessesEl.innerHTML = "";
    if (miniBarsEl) miniBarsEl.innerHTML = "";
    return;
  }
  if (!state.remainingAnswers.length) {
    bestGuessesEl.innerHTML = "";
    if (miniBarsEl) miniBarsEl.innerHTML = "";
    return;
  }

  const candidates = state.allowedWords;
  const candidateCodes = state.allowedCodes;
  const answers = state.remainingAnswers;
  const answerCodes = state.remainingCodes;
  const { weights, H0 } = getCurrentWeightsAndEntropy();
  state.currentWeights = weights;
  state.currentEntropy = H0;
  updateStats();
  const weightMap = new Map();
  for (let i = 0; i < answers.length; i += 1) {
    weightMap.set(answers[i], weights[i]);
  }
  const guessProbs = candidates.map((word) => weightMap.get(word) || 0);
  const results = [];
  let index = 0;
  const buckets = new Float64Array(PATTERN_COUNT);

  if (progressEl) {
    progressEl.hidden = false;
    progressBarEl.style.width = "0%";
  }

  function step() {
    const start = performance.now();
    while (index < candidates.length && performance.now() - start < 16) {
      buckets.fill(0);
      const guessCodes = candidateCodes[index];
      for (let i = 0; i < answerCodes.length; i += 1) {
        const code = patternCodeFromCodes(guessCodes, answerCodes[i]);
        buckets[code] += weights[i];
      }
      const H1 = entropyFromDistribution(buckets);
      const gain = H0 - H1;
      const prob = guessProbs[index] || 0;
      const expected = prob + (1 - prob) * (1 + entropyToExpectedScore(gain));
      results.push({ guess: candidates[index], score: expected, gain });
      index += 1;
    }
    const pct = (index / candidates.length) * 100;
    if (progressBarEl) progressBarEl.style.width = `${pct.toFixed(1)}%`;

    if (index < candidates.length) {
      requestAnimationFrame(step);
    } else {
      if (progressEl) progressEl.hidden = true;
      results.sort((a, b) => a.score - b.score);
      renderBestGuesses(results.slice(0, 10));
      renderMiniBars(results.slice(0, 2));
    }
  }

  requestAnimationFrame(step);
}

function renderBestGuesses(list) {
  bestGuessesEl.innerHTML = "";
  list.forEach((item) => {
    const li = document.createElement("li");
    const word = document.createElement("strong");
    word.textContent = item.guess.toLowerCase();
    const score = document.createElement("span");
    score.textContent = formatBits(item.gain);
    li.appendChild(word);
    li.appendChild(score);
    bestGuessesEl.appendChild(li);
  });
}
function renderMiniBars(list) {
  if (!miniBarsEl) return;
  miniBarsEl.innerHTML = "";
  if (!list.length) return;
  const maxGain = list[0].gain || 1;
  list.forEach((item) => {
    const row = document.createElement("div");
    row.className = "mini-bar";
    const label = document.createElement("span");
    label.textContent = item.guess.toLowerCase();
    const track = document.createElement("div");
    track.className = "mini-bar__track";
    const fill = document.createElement("div");
    fill.className = "mini-bar__fill";
    const ratio = Math.max(0, Math.min(1, item.gain / maxGain));
    fill.style.width = `${ratio * 100}%`;
    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    miniBarsEl.appendChild(row);
  });
}

function toggleMode() {
  state.mode = state.mode === "solver" ? "play" : "solver";
  toggleModeButton.textContent = state.mode === "solver" ? "Switch to Play" : "Switch to Solver";
  resetGame();
}

async function loadLocalWordList(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("Missing word list.");
  const text = await response.text();
  const words = text
    .split(/\s+/)
    .map((word) => word.trim().toLowerCase())
    .filter((word) => word.length === WORD_LENGTH);
  if (!words.length) throw new Error("Empty word list.");
  return words;
}

async function loadFreqMap(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("Missing frequency map.");
  return await response.json();
}

async function init() {
  setControlsEnabled(false);
  let allowedWords = [];
  let answerWords = [];
  let freqMap = null;
  try {
    [allowedWords, answerWords, freqMap] = await Promise.all([
      loadLocalWordList(LOCAL_ALLOWED),
      loadLocalWordList(LOCAL_ANSWERS),
      loadFreqMap(LOCAL_FREQ_MAP)
    ]);
  } catch (error) {
    patternHintEl.textContent =
      "Official lists not found. Add data/allowed.txt, data/answers.txt, and data/freq_map.json.";
    setControlsEnabled(false);
    return;
  }

  const allowedSet = new Set(allowedWords);
  answerWords.forEach((word) => allowedSet.add(word));
  state.allowedWords = Array.from(allowedSet);
  state.answerWords = answerWords;
  state.allowedIndex = new Map(state.allowedWords.map((word, idx) => [word, idx]));
  state.answerIndex = new Map(state.answerWords.map((word, idx) => [word, idx]));
  state.allowedCodes = state.allowedWords.map(wordToCodes);
  state.answerCodes = state.answerWords.map(wordToCodes);
  state.remainingAnswers = [...state.answerWords];
  state.remainingCodes = state.answerCodes.slice();
  if (state.allowedWords.length < MIN_ALLOWED_SIZE || state.answerWords.length < MIN_ANSWER_SIZE) {
    patternHintEl.textContent =
      "Lists are too small. Replace data/allowed.txt and data/answers.txt with official lists.";
    setControlsEnabled(false);
    return;
  }

  state.priors = buildFrequencyPriors(freqMap);
  state.listsReady = true;

  buildBoard();
  buildKeyboard();
  setSecret();
  updateStats();
  computeSuggestions();
  applyKeyboardStyles();
  setControlsEnabled(true);
}

boardEl.addEventListener("click", handleTileClick);
enterButton.addEventListener("click", handleEnter);
applyButton.addEventListener("click", handleApply);
resetButton.addEventListener("click", resetGame);
toggleModeButton.addEventListener("click", toggleMode);
randomAnswerButton.addEventListener("click", () => {
  setSecret();
  if (state.mode === "play") {
    resetGame();
  }
});
revealAnswerButton.addEventListener("click", revealSecret);

keyboardEl.addEventListener("click", (event) => {
  const key = event.target.closest(".key");
  if (!key) return;
  const code = key.dataset.key;
  if (code === "enter") {
    handleEnter();
  } else if (code === "backspace") {
    guessInput.value = guessInput.value.slice(0, -1);
    updateRowLetters(state.currentRow, sanitizeGuess(guessInput.value));
  } else if (code && code.length === 1) {
    if (guessInput.value.length < WORD_LENGTH && state.pendingRow === null) {
      guessInput.value += code;
      updateRowLetters(state.currentRow, sanitizeGuess(guessInput.value));
    }
  }
});

guessInput.addEventListener("input", (event) => {
  const value = sanitizeGuess(event.target.value);
  guessInput.value = value;
  if (state.pendingRow === null) updateRowLetters(state.currentRow, value);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    handleEnter();
  } else if (event.key === "Backspace") {
    guessInput.value = guessInput.value.slice(0, -1);
    updateRowLetters(state.currentRow, sanitizeGuess(guessInput.value));
  }
});

init();
