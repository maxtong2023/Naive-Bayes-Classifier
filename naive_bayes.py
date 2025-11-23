import math
import re
from collections import defaultdict


class Bayes_Classifier:
    """Naïve Bayes classifier for binary sentiment labels (1 vs 5 stars)."""

    def __init__(self, alpha=1.0, useBigrams=True):
        self.alphaVal  = alpha
        self.useBigrams = useBigrams
        self.labels = ('1', '5')
        self.stopWords = self._buildStopwords()
        self._resetModel()

    def _resetModel(self):
        self.wordCounts = {label: defaultdict(int) for label in self.labels}
        self.totalWordCounts = {label: 0 for label in self.labels}
        self.docCounts = {label: 0 for label in self.labels}
        self.totalDocs = 0
        self.vocabSet = set()
        self.logPriors = {label: float('-inf') for label in self.labels}
        self.vocabSize = 1

    def train(self, lines):
        """Train the classifier with labeled reviews."""
        self._resetModel()
        for rawLine in lines:
            if not rawLine:
                continue
            line = rawLine.strip()  
            if line == '':
                continue
            pieces = line.split('|', 2)
            if len(pieces) != 3:
                continue
            label = pieces[0]
            reviewText = pieces[2]
            if label not in self.wordCounts:
                continue

            feats = self._extractFeatures(reviewText)
            self.docCounts[label] += 1
            self.totalDocs += 1

            for token in feats:
                self.wordCounts[label][token] += 1
                self.totalWordCounts[label] += 1
                self.vocabSet.add(token)

        if len(self.vocabSet) > 0:
            self.vocabSize = len(self.vocabSet)

        totalLabels = len(self.labels)
        priorDenom = self.totalDocs + totalLabels * self.alphaVal
        if priorDenom == 0:
            priorDenom = 1

        for label in self.labels:
            numerator = self.docCounts[label] + self.alphaVal
            chance = numerator / priorDenom
            if chance <= 0:
                chance = 1e-12
            self.logPriors[label] = math.log(chance)

    def classify(self, lines):
        """Classify new reviews; returns list of predicted labels."""
        predictions = []
        denominatorCache = {}
        for label in self.labels:
            denominatorCache[label] = self.totalWordCounts[label] + self.alphaVal * self.vocabSize

        for rawLine in lines:
            line = rawLine.strip()
            parts = line.split('|', 2)
            if len(parts) == 3:
                text = parts[2]
            else:
                text = ''
            feats = self._extractFeatures(text)

            scoreCard = {}
            for label in self.labels:
                if self.docCounts[label] == 0:
                    scoreCard[label] = float('-inf')
                    continue
                logScore = self.logPriors[label]
                denom = denominatorCache[label]
                if denom == 0:
                    denom = 1
                for token in feats:
                    count = self.wordCounts[label].get(token, 0)
                    value = (count + self.alphaVal) / denom
                    if value <= 0:
                        value = 1e-12
                    logScore += math.log(value)
                scoreCard[label] = logScore

            bestLabel = '1'
            bestVal = scoreCard.get(bestLabel, float('-inf'))
            for label in self.labels:
                if scoreCard.get(label, float('-inf')) > bestVal:
                    bestVal = scoreCard[label]
                    bestLabel = label
            predictions.append(bestLabel)

        return predictions

    def _extractFeatures(self, text):
        tokens = self._tokenizeText(text)
        features = []
        for token in tokens:
            if token:
                features.append(token)
        if self.useBigrams and len(tokens) > 1:
            idx = 0
            while idx < len(tokens) - 1:
                bigram = tokens[idx] + '_' + tokens[idx + 1]
                features.append(bigram)
                idx += 1
        return features

    def _tokenizeText(self, text):
        lowered = text.lower()
        cleaned = re.sub(r"[^a-z0-9\s']", "  ", lowered)
        rawTokens = cleaned.split()
        filtered = []
        for tok in rawTokens:
            if tok not in self.stopWords:
                filtered.append(self._stemWord(tok))
        return filtered

    def _stemWord(self, word):
        if len(word) <= 3:
            return word

        saved = word
        endings = (
            'ational', 'fulness', 'iveness', 'ousness', 'ization',
            'biliti', 'tional', 'lessli', 'entli', 'ation', 'alism',
            'aliti', 'ousli', 'fulli', 'enci', 'anci', 'abli',
            'izer', 'ator', 'logi', 'ical', 'ness', 'ment', 'ingly',
            'edly', 'ing', 'ers', 'ies', 'ied', 'ly', 'ed', 'es', 's'
        )
        for ending in endings:
            if word.endswith(ending) and len(word) - len(ending) >= 3:
                word = word[:-len(ending)]
                break

        if word.endswith('er') and len(word) > 4:
            word = word[:-2]
        if word.endswith('ly') and len(word) > 4:
            word = word[:-2]
        if word.endswith('e') and len(word) > 3:
            word = word[:-1]
        if not word:
            word = saved
        return word

    def _buildStopwords(self):
        base = """
        a about above after again against all am an and any are aren't as at
        be because been before being below between both but by
        can can't cannot could couldn't did didn't do does doesn't doing don't down during
        each few for from further
        had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's
        i i'd i'll i'm i've if in into is isn't it it's its itself
        let's me more most mustn't my myself
        no nor not of off on once only or other ought our ours ourselves out over own
        same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very
        was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
        """
        temp = set()
        for part in base.split():
            cleaned = part.strip().lower()
            if cleaned:
                temp.add(cleaned)
        return frozenset(temp)
