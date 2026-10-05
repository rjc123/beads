package main

import (
	"fmt"
	"sort"
	"testing"

	"github.com/steveyegge/beads/internal/types"
)

// rankingCorpus builds a backlog with the shape that breaks unweighted
// similarity scoring: many short, semantically unrelated chores that share an
// imperative verb and an identical captured-from-triage description, plus one
// genuine duplicate pair that shares a distinctive subject but almost none of
// its wording.
func rankingCorpus() (issues []*types.Issue, trueA, trueB, falseA, falseB string) {
	const boilerplate = "Captured during the weekly review. No further detail recorded yet."

	items := []string{
		"draught excluder", "mitre block", "garden twine", "door wedge", "hose reel",
		"bike pump", "tile grout", "curtain pole", "sink plunger", "shelf bracket",
		"wall anchor", "cable clip", "drawer runner", "window catch", "fence post",
		"loft hatch", "step ladder", "putty knife", "dust sheet", "socket set",
		"gaffer tape", "bin liner", "radiator key", "chimney brush", "gutter scoop",
		"pipe lagging", "sash cord", "latch plate", "hinge pin", "trellis panel",
		"mortar board", "spirit level", "chalk line", "rasp file", "awning crank",
		"valve seal", "skirting trim", "lintel strap", "joist hanger", "newel cap",
	}
	rooms := []string{"office", "stairway", "garden", "kitchen", "garage", "hallway", "loft", "shed"}

	// Every chore names a different item, so no two of them are near-duplicates
	// of each other on the merits. All they share is the verb and the template.
	for i, item := range items {
		issues = append(issues, &types.Issue{
			ID:          fmt.Sprintf("chore-%03d", i+1),
			Title:       fmt.Sprintf("Purchase %s for the %s", item, rooms[i%len(rooms)]),
			Description: boilerplate,
		})
	}

	// The two chores the unweighted metric scores highest: no shared subject at
	// all, only the shared verb and the shared description template.
	falseA, falseB = "chore-001", "chore-003"

	issues = append(issues,
		&types.Issue{
			ID:    "dup-a",
			Title: "Stream audio from an Android phone to the AirPlay speakers",
			Description: "Investigate whether the household speaker set can be driven straight " +
				"from a handset. The receivers are an AirPort Express plus two powered monitors. " +
				"Needs a protocol check and a latency measurement first.",
		},
		&types.Issue{
			ID:    "dup-b",
			Title: "Can an Android handset play music through AirPlay 1 and 2 receivers?",
			Description: "Work out which AirPlay revisions the AirPort Express understands, and " +
				"whether an Android device is able to act as a sender at all. Record what is " +
				"found, then close.",
		},
	)
	return issues, "dup-a", "dup-b", falseA, falseB
}

// rankOf returns the 1-based position of the (a, b) pair in the scored output,
// and its similarity. A rank of 0 means the pair was not scored at all.
func rankOf(pairs []duplicatePair, a, b string) (int, float64) {
	sort.SliceStable(pairs, func(i, j int) bool { return pairs[i].Similarity > pairs[j].Similarity })
	for i, p := range pairs {
		if (p.IssueA.ID == a && p.IssueB.ID == b) || (p.IssueA.ID == b && p.IssueB.ID == a) {
			return i + 1, p.Similarity
		}
	}
	return 0, 0
}

// TestFindMechanicalDuplicatesRanksRareTokenOverlapFirst is the regression test
// for the defect in gastownhall/beads#6075: on a real backlog the unweighted
// metric ranked a true duplicate pair below thousands of false ones, so no
// threshold could recover it. The assertion is therefore about RANK, not about
// how many pairs come back.
func TestFindMechanicalDuplicatesRanksRareTokenOverlapFirst(t *testing.T) {
	issues, trueA, trueB, falseA, falseB := rankingCorpus()

	// Threshold 0 so that every pair is scored and ranking is the only variable.
	rawPairs := findMechanicalDuplicatesWeighted(issues, 0, weightingRaw)
	idfPairs := findMechanicalDuplicatesWeighted(issues, 0, weightingIDF)

	rawTrueRank, rawTrueScore := rankOf(rawPairs, trueA, trueB)
	rawFalseRank, rawFalseScore := rankOf(rawPairs, falseA, falseB)
	idfTrueRank, idfTrueScore := rankOf(idfPairs, trueA, trueB)
	idfFalseRank, idfFalseScore := rankOf(idfPairs, falseA, falseB)

	t.Logf("raw: true pair rank %d (%.4f), unrelated-chore pair rank %d (%.4f)",
		rawTrueRank, rawTrueScore, rawFalseRank, rawFalseScore)
	t.Logf("idf: true pair rank %d (%.4f), unrelated-chore pair rank %d (%.4f)",
		idfTrueRank, idfTrueScore, idfFalseRank, idfFalseScore)

	// Before: the true pair is buried. The exact depth depends on the corpus
	// size, so assert only that it is far from the top and below the chore pair.
	if rawTrueRank <= 100 {
		t.Errorf("unweighted scoring was expected to bury the true pair, but ranked it %d", rawTrueRank)
	}
	if rawTrueScore >= rawFalseScore {
		t.Errorf("unweighted scoring was expected to rank two unrelated chores (%.4f) above the true duplicate (%.4f)",
			rawFalseScore, rawTrueScore)
	}

	// After: rare-token weighting puts the true pair first, and demotes the
	// chore pair below it.
	if idfTrueRank != 1 {
		t.Errorf("IDF weighting ranked the true duplicate pair %d, want 1", idfTrueRank)
	}
	if idfFalseScore >= idfTrueScore {
		t.Errorf("IDF weighting scored two unrelated chores %.4f, not below the true duplicate's %.4f",
			idfFalseScore, idfTrueScore)
	}
	// The chore pair is the archetypal false positive from the bug report: two
	// unrelated errands scoring above the default threshold on shared verb and
	// shared template alone. It must end up below that threshold.
	const defaultThreshold = 0.5
	if rawFalseScore < defaultThreshold {
		t.Fatalf("test corpus no longer reproduces the defect: unrelated chores scored %.4f under unweighted scoring, want >= %.2f",
			rawFalseScore, defaultThreshold)
	}
	if idfFalseScore >= defaultThreshold {
		t.Errorf("IDF weighting still scores two unrelated chores %.4f, want below the default threshold %.2f",
			idfFalseScore, defaultThreshold)
	}
	_ = idfFalseRank
}

// TestWeightedSimilarityReducesToUnweighted pins the compatibility property the
// --weighting raw escape hatch depends on: nil weights must reproduce the
// original scoring exactly.
func TestWeightedSimilarityReducesToUnweighted(t *testing.T) {
	a := map[string]int{"fix": 2, "auth": 1, "bug": 1}
	b := map[string]int{"fix": 1, "auth": 3, "login": 1}

	unit := tokenWeights{"fix": 1, "auth": 1, "bug": 1, "login": 1}

	if got, want := weightedJaccardSimilarity(a, b, unit), jaccardSimilarity(a, b); got != want {
		t.Errorf("weighted Jaccard with unit weights = %v, want %v", got, want)
	}
	if got, want := weightedCosineSimilarity(a, b, unit), cosineSimilarity(a, b); got != want {
		t.Errorf("weighted cosine with unit weights = %v, want %v", got, want)
	}
}

// TestInverseDocumentFrequenciesZeroesUbiquitousTokens pins the property the
// whole fix rests on: a token every issue carries cannot contribute to a score.
func TestInverseDocumentFrequenciesZeroesUbiquitousTokens(t *testing.T) {
	docs := []map[string]int{
		{"the": 1, "alpha": 1},
		{"the": 1, "beta": 1},
		{"the": 1, "gamma": 1},
	}
	weights := inverseDocumentFrequencies(docs)

	if w := weights.weight("the"); w != 0 {
		t.Errorf("token present in every document weighs %v, want 0", w)
	}
	if w := weights.weight("alpha"); w <= 0 {
		t.Errorf("token present in one document of three weighs %v, want > 0", w)
	}

	// Two issues whose only shared token is ubiquitous must score zero.
	if s := pairSimilarity(docs[0], docs[1], weights); s != 0 {
		t.Errorf("pair sharing only a ubiquitous token scored %v, want 0", s)
	}
}
