const fs = require("fs");
const path = require("path");

const WORD_LENGTH = 5;
const PATTERN_COUNT = 3 ** WORD_LENGTH;
const ALL_GREEN_CODE = 242;

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        args[key] = next;
        i += 1;
      } else {
        args[key] = true;
      }
    }
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const guessSetMode = args["guess-set"] || "answers"; // allowed | answers
const priorMode = args.prior || "uniform"; // uniform | freq
const entropyCoef = Number.isFinite(Number(args.coef)) ? Number(args.coef) : 1.5;
const sampleSize = Number.isFinite(Number(args.sample)) ? Number(args.sample) : 0;
const seed = Number.isFinite(Number(args.seed)) ? Number(args.seed) : 1;
const depth = Number.isFinite(Number(args.depth)) ? Number(args.depth) : 1;
const deepThreshold = Number.isFinite(Number(args["deep-threshold"]))
  ? Number(args["deep-threshold"])
  : 80;
const deepCandidateMode = args["deep-candidates"] || "answers"; // answers | all

function loadWordList(filePath) {
  const text = fs.readFileSync(filePath, "utf8");
  return text
    .split(/\s+/)
    .map((word) => word.trim().toLowerCase())
    .filter((word) => word.length === WORD_LENGTH);
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
    if (p > 0) entropy -= p * Math.log2(p);
  }
  return entropy;
}

function entropyToExpectedScore(entropy) {
  const twoToMinus = 2 ** -entropy;
  const minScore = twoToMinus + 2 * (1 - twoToMinus);
  return minScore + (entropyCoef * entropy) / 11.5;
}

function wordToCodes(word) {
  const codes = new Uint8Array(WORD_LENGTH);
  for (let i = 0; i < WORD_LENGTH; i += 1) {
    codes[i] = word.charCodeAt(i) - 97;
  }
  return codes;
}

const scratchCounts = new Uint8Array(26);
const scratchResult = new Uint8Array(WORD_LENGTH);

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

function buildPatternTable(candidates, candidateCodes, answerCodes) {
  const answerCount = answerCodes.length;
  const table = new Uint8Array(candidates.length * answerCount);
  for (let i = 0; i < candidates.length; i += 1) {
    const guessCodes = candidateCodes[i];
    const offset = i * answerCount;
    for (let j = 0; j < answerCount; j += 1) {
      table[offset + j] = patternCodeFromCodes(guessCodes, answerCodes[j]);
    }
    if ((i + 1) % 200 === 0) {
      process.stdout.write(`\rBuilding pattern table ${i + 1}/${candidates.length}`);
    }
  }
  process.stdout.write("\n");
  return table;
}

function getRemainingWeights(remainingIndices, answerWords, priors) {
  const weights = new Float64Array(remainingIndices.length);
  let total = 0;
  for (let i = 0; i < remainingIndices.length; i += 1) {
    const word = answerWords[remainingIndices[i]];
    const weight = priors?.get(word) ?? 1;
    weights[i] = weight;
    total += weight;
  }
  if (!total) {
    const uniform = 1 / (remainingIndices.length || 1);
    weights.fill(uniform);
    return weights;
  }
  for (let i = 0; i < weights.length; i += 1) {
    weights[i] /= total;
  }
  return weights;
}

function filterRemaining(table, guessIndex, remainingIndices, answerCount, patternCode) {
  const nextRemaining = [];
  const offset = guessIndex * answerCount;
  for (let i = 0; i < remainingIndices.length; i += 1) {
    const answerIndex = remainingIndices[i];
    const code = table[offset + answerIndex];
    if (code === patternCode) {
      nextRemaining.push(answerIndex);
    }
  }
  return nextRemaining;
}

function bestScoreDepth1({
  remainingIndices,
  candidateIndices,
  answerWords,
  answerCount,
  table,
  priors,
  candidateAnswerIndex
}) {
  if (remainingIndices.length <= 1) {
    return remainingIndices.length ? 1 : 0;
  }
  const weights = getRemainingWeights(remainingIndices, answerWords, priors);
  const H0 = entropyFromDistribution(weights);
  const weightLookup = new Float64Array(answerCount);
  for (let i = 0; i < remainingIndices.length; i += 1) {
    weightLookup[remainingIndices[i]] = weights[i];
  }
  let bestScore = Infinity;
  const buckets = new Float64Array(PATTERN_COUNT);

  for (let i = 0; i < candidateIndices.length; i += 1) {
    const candIndex = candidateIndices[i];
    buckets.fill(0);
    const offset = candIndex * answerCount;
    for (let j = 0; j < remainingIndices.length; j += 1) {
      const patternCode = table[offset + remainingIndices[j]];
      buckets[patternCode] += weights[j];
    }
    const H1 = entropyFromDistribution(buckets);
    const gain = H0 - H1;
    const answerIndex = candidateAnswerIndex[candIndex];
    const prob = answerIndex >= 0 ? weightLookup[answerIndex] : 0;
    const expected = prob + (1 - prob) * (1 + entropyToExpectedScore(gain));
    if (expected < bestScore) bestScore = expected;
  }
  return bestScore;
}

function pickBestGuessDepth1({
  remainingIndices,
  candidateIndices,
  answerWords,
  answerCount,
  table,
  priors,
  candidateAnswerIndex
}) {
  const weights = getRemainingWeights(remainingIndices, answerWords, priors);
  const H0 = entropyFromDistribution(weights);
  const weightLookup = new Float64Array(answerCount);
  for (let i = 0; i < remainingIndices.length; i += 1) {
    weightLookup[remainingIndices[i]] = weights[i];
  }
  let bestGuessIndex = candidateIndices[0];
  let bestScore = Infinity;
  const buckets = new Float64Array(PATTERN_COUNT);

  for (let i = 0; i < candidateIndices.length; i += 1) {
    const candIndex = candidateIndices[i];
    buckets.fill(0);
    const offset = candIndex * answerCount;
    for (let j = 0; j < remainingIndices.length; j += 1) {
      const patternCode = table[offset + remainingIndices[j]];
      buckets[patternCode] += weights[j];
    }
    const H1 = entropyFromDistribution(buckets);
    const gain = H0 - H1;
    const answerIndex = candidateAnswerIndex[candIndex];
    const prob = answerIndex >= 0 ? weightLookup[answerIndex] : 0;
    const expected = prob + (1 - prob) * (1 + entropyToExpectedScore(gain));
    if (expected < bestScore) {
      bestScore = expected;
      bestGuessIndex = candIndex;
    }
  }
  return { bestGuessIndex, bestScore };
}

function pickBestGuessDepth2({
  remainingIndices,
  candidateIndices,
  answerWords,
  answerCount,
  table,
  priors,
  candidateAnswerIndex,
  answerToCandidateIndex,
  memo
}) {
  const weights = getRemainingWeights(remainingIndices, answerWords, priors);
  const weightLookup = new Float64Array(answerCount);
  for (let i = 0; i < remainingIndices.length; i += 1) {
    weightLookup[remainingIndices[i]] = weights[i];
  }
  let bestGuessIndex = candidateIndices[0];
  let bestScore = Infinity;
  const bucketsWeight = new Float64Array(PATTERN_COUNT);
  const bucketIndices = Array.from({ length: PATTERN_COUNT }, () => []);

  for (let i = 0; i < candidateIndices.length; i += 1) {
    const candIndex = candidateIndices[i];
    bucketsWeight.fill(0);
    for (let p = 0; p < PATTERN_COUNT; p += 1) bucketIndices[p].length = 0;
    const offset = candIndex * answerCount;
    for (let j = 0; j < remainingIndices.length; j += 1) {
      const answerIndex = remainingIndices[j];
      const patternCode = table[offset + answerIndex];
      bucketsWeight[patternCode] += weights[j];
      if (patternCode !== ALL_GREEN_CODE) bucketIndices[patternCode].push(answerIndex);
    }
    let expectedNext = 0;
    for (let code = 0; code < PATTERN_COUNT; code += 1) {
      const p = bucketsWeight[code];
      if (!p) continue;
      if (code === ALL_GREEN_CODE) continue;
      const subset = bucketIndices[code];
      if (!subset.length) continue;
      let nextScore;
      if (subset.length === 1) {
        nextScore = 1;
      } else {
        const key = subset.join(",");
        if (memo.has(key)) {
          nextScore = memo.get(key);
        } else {
          nextScore = bestScoreDepth1({
            remainingIndices: subset,
            candidateIndices: answerToCandidateIndex
              ? subset
                  .map((idx) => answerToCandidateIndex[idx])
                  .filter((value) => Number.isInteger(value))
              : subset,
            answerWords,
            answerCount,
            table,
            priors,
            candidateAnswerIndex
          });
          memo.set(key, nextScore);
        }
      }
      expectedNext += p * nextScore;
    }
    const totalScore = 1 + expectedNext;
    if (totalScore < bestScore) {
      bestScore = totalScore;
      bestGuessIndex = candIndex;
    }
  }
  return { bestGuessIndex, bestScore };
}

function lcg(seedValue) {
  let state = seedValue >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

function shuffle(arr, seedValue) {
  const rand = lcg(seedValue);
  const copy = arr.slice();
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function simulateAll({
  candidates,
  candidateIndexMap,
  candidateAnswerIndex,
  answerWords,
  answerIndexMap,
  priors,
  sample,
  table,
  answerToCandidateIndex
}) {
  let totalGuesses = 0;
  let totalWeightedGuesses = 0;
  let totalWeight = 0;
  let failures = 0;
  const sampleWeights = getWeights(sample, priors);
  const answerCount = answerWords.length;
  const memo = new Map();

  for (let idx = 0; idx < sample.length; idx += 1) {
    const answer = sample[idx];
    const answerIndex = answerIndexMap.get(answer);
    let remainingIndices = Array.from({ length: answerCount }, (_, i) => i);
    let guesses = 0;

    while (guesses < 6) {
      const useDepth2 = depth >= 2 && remainingIndices.length <= deepThreshold;
      const candidateIndices =
        useDepth2 && deepCandidateMode === "answers"
          ? remainingIndices
              .map((answerIdx) => candidateIndexMap.get(answerWords[answerIdx]))
              .filter((value) => Number.isInteger(value))
          : candidates.map((_, i) => i);
      if (!candidateIndices.length) {
        candidateIndices.push(...candidates.map((_, i) => i));
      }
      const result = useDepth2
        ? pickBestGuessDepth2({
            remainingIndices,
            candidateIndices,
            answerWords,
            answerCount,
            table,
            priors,
            candidateAnswerIndex,
            answerToCandidateIndex,
            memo
          })
        : pickBestGuessDepth1({
            remainingIndices,
            candidateIndices,
            answerWords,
            answerCount,
            table,
            priors,
            candidateAnswerIndex
          });
      const guessIndex = result.bestGuessIndex;
      guesses += 1;
      if (candidateAnswerIndex[guessIndex] === answerIndex) break;
      const patternCode = table[guessIndex * answerCount + answerIndex];
      remainingIndices = filterRemaining(table, guessIndex, remainingIndices, answerCount, patternCode);
      if (!remainingIndices.length) break;
    }

    if (guesses > 6) failures += 1;
    totalGuesses += guesses;
    const weight = sampleWeights[idx] ?? 0;
    totalWeightedGuesses += guesses * weight;
    totalWeight += weight;

    if ((idx + 1) % 20 === 0) {
      process.stdout.write(`\rProcessed ${idx + 1}/${sample.length}`);
    }
  }
  process.stdout.write("\n");
  return {
    average: totalGuesses / sample.length,
    weightedAverage: totalWeight ? totalWeightedGuesses / totalWeight : totalGuesses / sample.length,
    total: totalGuesses,
    failures
  };
}

function main() {
  const dataDir = path.resolve(__dirname, "..", "data");
  const allowedWords = loadWordList(path.join(dataDir, "allowed.txt"));
  const answerWords = loadWordList(path.join(dataDir, "answers.txt"));
  const freqMap = JSON.parse(fs.readFileSync(path.join(dataDir, "freq_map.json"), "utf8"));

  const candidates = guessSetMode === "answers" ? answerWords : Array.from(new Set([...allowedWords, ...answerWords]));
  const candidateCodes = candidates.map(wordToCodes);
  const answerCodes = answerWords.map(wordToCodes);
  const candidateIndexMap = new Map(candidates.map((word, idx) => [word, idx]));
  const answerIndexMap = new Map(answerWords.map((word, idx) => [word, idx]));
  const candidateAnswerIndex = candidates.map((word) =>
    answerIndexMap.has(word) ? answerIndexMap.get(word) : -1
  );
  const answerToCandidateIndex = answerWords.map((word) =>
    candidateIndexMap.has(word) ? candidateIndexMap.get(word) : -1
  );

  let priors = null;
  if (priorMode === "freq") {
    priors = buildFrequencyPriors(freqMap);
  }

  let sample = answerWords;
  if (sampleSize && sampleSize > 0 && sampleSize < answerWords.length) {
    sample = shuffle(answerWords, seed).slice(0, sampleSize);
  }

  console.log(`Guess set: ${guessSetMode}`);
  console.log(`Prior mode: ${priorMode}`);
  console.log(`Entropy coef: ${entropyCoef}`);
  console.log(`Depth: ${depth}`);
  if (depth >= 2) {
    console.log(`Deep threshold: ${deepThreshold}`);
    console.log(`Deep candidates: ${deepCandidateMode}`);
  }
  console.log(`Answers: ${sample.length}`);
  console.log("Building pattern table...");
  const table = buildPatternTable(candidates, candidateCodes, answerCodes);

  const results = simulateAll({
    candidates,
    candidateIndexMap,
    candidateAnswerIndex,
    answerWords,
    answerIndexMap,
    priors,
    sample,
    table,
    answerToCandidateIndex
  });

  console.log(`Average guesses (uniform): ${results.average.toFixed(3)}`);
  console.log(`Average guesses (weighted): ${results.weightedAverage.toFixed(3)}`);
  console.log(`Total guesses: ${results.total}`);
  console.log(`Failures: ${results.failures}`);
}

main();
