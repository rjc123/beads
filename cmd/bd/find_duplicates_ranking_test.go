package main

import (
	"fmt"
	"sort"
	"testing"

	"github.com/steveyegge/beads/internal/types"
)

// rankingCorpus builds a backlog with the two shapes that break naive scoring
// on a real tracker:
//
//   - CHORES: many short, unrelated errands sharing an imperative verb and one
//     identical captured-from-triage description. These defeat unweighted
//     scoring, which is dominated by common-token overlap.
//   - A TEMPLATE FAMILY: issues whose titles have nothing whatever in common
//     but whose descriptions share a generated skeleton. These defeat IDF
//     scoring over a single flat bag of words, because a skeleton used by only
//     part of the corpus is globally rare and so scores as strong evidence —
//     and the description is far longer than the title, so it wins.
//
// Plus one genuine duplicate pair, which shares a distinctive subject in its
// title and almost none of its wording anywhere else.
func rankingCorpus() (issues []*types.Issue, trueA, trueB, choreA, choreB, tmplA, tmplB string) {
	const boilerplate = "Captured during the weekly review. No further detail recorded yet."

	// A realistic errand population: real trackers vary the verb, so two
	// unrelated chores share little more than a preposition and the template.
	// The corpus is deliberately large enough for document frequencies to carry
	// information — the defect being fixed is a corpus-statistics problem, and a
	// couple of dozen issues cannot exhibit it.
	verbs := []string{"Purchase", "Buy", "Order", "Source", "Replace", "Collect", "Fit", "Repair"}
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
	rooms := []string{
		"office", "stairway", "garden", "kitchen", "garage",
		"hallway", "loft", "shed", "landing", "porch",
	}
	for i, item := range items {
		for r := 0; r < 5; r++ {
			n := len(issues) + 1
			issues = append(issues, &types.Issue{
				ID:          fmt.Sprintf("chore-%03d", n),
				Title:       fmt.Sprintf("%s %s for the %s", verbs[n%len(verbs)], item, rooms[(i+r)%len(rooms)]),
				Description: boilerplate,
			})
		}
	}
	// Two chores with different verbs, different items and different rooms:
	// unrelated in every way except the template and the word "for". They must
	// not share an item — five consecutive entries reuse one — so these are ten
	// apart.
	choreA, choreB = "chore-001", "chore-011"

	// The template family: unrelated titles, near-identical long descriptions.
	steps := []string{
		"Set the venture's goals", "Inventory every step the build requires",
		"Choose the tool stack", "Define the target structure",
		"Build the working prototype", "Draft the launch note",
		"Measure the first cohort", "Retire the scaffolding",
		"Agree the pricing ladder", "Write the handover pack",
		"Pick the hosting region", "Record the baseline numbers",
		"Name the first segment", "Schedule the review cadence",
		"Close the discovery phase", "Publish the internal brief",
		"Rehearse the demo script", "Sign off the budget line",
		"Archive the spike notes", "Hand back the shared licence",
	}
	for i, step := range steps {
		issues = append(issues, &types.Issue{
			ID:    fmt.Sprintf("tmpl-%03d", i+1),
			Title: step,
			Description: fmt.Sprintf("Child of the venture build framework. Framework step W%d, "+
				"stage SETUP, sub-stage Plan Mission. The four steps of every framework child are "+
				"the same: establish what the step is for, find or build the capability behind it, "+
				"prove that capability against a real venture, and close with a reusable thing. "+
				"Acceptance is the reusable thing existing and being referenced elsewhere.", i+1),
		})
	}
	tmplA, tmplB = "tmpl-001", "tmpl-003"

	// A background population that makes the fixture's vocabulary behave like a
	// real backlog's. Without it every generic word in the true pair's titles
	// ("stream", "audio", "music", "speakers") would be unique and therefore
	// carry maximal IDF, inflating the union and burying the very pair the fix
	// is meant to surface. Real trackers mention these words constantly; only
	// the subject of a duplicate is rare.
	generic := []string{
		"Stream the build log to the dashboard",
		"Audio drops out during the weekly call",
		"Phone number formatting is wrong on export",
		"Music licence renewal for the intro clip",
		"Play the onboarding video automatically",
		"Device inventory is out of date",
		"Speakers for the meeting room are unassigned",
		"Receivers are not acknowledging the webhook",
		"Express delivery option missing at checkout",
		"Handset provisioning takes three days",
		"Stream the metrics into the warehouse",
		"Audio normalisation for podcast uploads",
		"Phone tree routes to the wrong queue",
		"Music metadata import drops the album field",
		"Play-by-play replay for failed jobs",
		"Device pairing fails on first attempt",
		"Speakers list for the conference is stale",
		"Receivers need a retry budget",
		"Express lane for urgent tickets",
		"Handset battery reporting is inaccurate",
	}
	for i, title := range generic {
		issues = append(issues, &types.Issue{
			ID:          fmt.Sprintf("misc-%03d", i+1),
			Title:       title,
			Description: boilerplate,
		})
	}

	issues = append(issues,
		&types.Issue{
			ID:    "dup-a",
			Title: "Stream audio from an Android phone to the AirPlay speakers",
			Description: "Investigate whether the household speaker set can be driven straight " +
				"from a handset. The receivers are an AirPort Express plus two powered monitors.",
		},
		&types.Issue{
			ID:    "dup-b",
			Title: "Can an Android handset play music through AirPlay 1 and 2 receivers?",
			Description: "Work out which AirPlay revisions the AirPort Express understands, and " +
				"whether an Android device is able to act as a sender at all.",
		},
	)
	return issues, "dup-a", "dup-b", choreA, choreB, tmplA, tmplB
}

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
// for gastownhall/beads#6075: on a real backlog the metric ranked a true
// duplicate below thousands of false pairs, so no threshold could recover it.
// The assertions are about RANK, not about how many pairs come back, because
// "fewer pairs returned" is the easy wrong claim here.
func TestFindMechanicalDuplicatesRanksRareTokenOverlapFirst(t *testing.T) {
	issues, trueA, trueB, choreA, choreB, tmplA, tmplB := rankingCorpus()

	// Threshold 0 so every pair is scored and ranking is the only variable.
	raw := findMechanicalDuplicatesWeighted(issues, 0, weightingRaw)
	idf := findMechanicalDuplicatesWeighted(issues, 0, weightingIDF)

	rawTrueRank, rawTrue := rankOf(raw, trueA, trueB)
	rawChoreRank, rawChore := rankOf(raw, choreA, choreB)
	_, rawTmpl := rankOf(raw, tmplA, tmplB)
	idfTrueRank, idfTrue := rankOf(idf, trueA, trueB)
	_, idfChore := rankOf(idf, choreA, choreB)
	_, idfTmpl := rankOf(idf, tmplA, tmplB)

	t.Logf("raw: true rank %d (%.4f), chore pair %.4f (rank %d), template pair %.4f",
		rawTrueRank, rawTrue, rawChore, rawChoreRank, rawTmpl)
	t.Logf("idf: true rank %d (%.4f), chore pair %.4f, template pair %.4f",
		idfTrueRank, idfTrue, idfChore, idfTmpl)

	// The corpus must still reproduce the defect, or this test asserts nothing.
	// Both classes of false positive have to beat the true pair beforehand.
	if rawTrue >= rawChore {
		t.Fatalf("corpus no longer reproduces the defect: unweighted scoring put the true pair (%.4f) at or above two unrelated chores (%.4f)", rawTrue, rawChore)
	}
	if rawTrue >= rawTmpl {
		t.Fatalf("corpus no longer reproduces the defect: unweighted scoring put the true pair (%.4f) at or above two template siblings (%.4f)", rawTrue, rawTmpl)
	}
	if rawTrueRank <= 100 {
		t.Errorf("unweighted scoring was expected to bury the true pair; it ranked %d", rawTrueRank)
	}

	// After: both inversions are corrected. These are the orderings the fix
	// contracts for. Note what is deliberately NOT asserted — that the true pair
	// ranks first. This fixture contains hundreds of GENUINE near-duplicates by
	// construction (five chores share each item, differing only in room), and
	// those legitimately outrank a pair that shares a subject but little wording.
	// The effect at scale — the pair moving into the default output on a real
	// 2,693-issue backlog — is demonstrated by measurement, not by this fixture,
	// whose absolute scores depend on its own vocabulary.
	if idfTrue <= idfChore {
		t.Errorf("two unrelated chores scored %.4f, at or above the true duplicate's %.4f", idfChore, idfTrue)
	}
	if idfTrue <= idfTmpl {
		t.Errorf("two template siblings with unrelated titles scored %.4f, at or above the true duplicate's %.4f; a shared description template is still outvoting the title", idfTmpl, idfTrue)
	}
	if idfTrueRank >= rawTrueRank {
		t.Errorf("the true pair ranked %d after and %d before; the fix must move it up", idfTrueRank, rawTrueRank)
	}

	// And both false classes must fall below the default threshold, so neither is
	// reported at all by a default invocation.
	const defaultThreshold = 0.5
	if rawChore < defaultThreshold {
		t.Fatalf("corpus no longer reproduces the defect: unrelated chores scored %.4f, below the default threshold, so they were never reported anyway", rawChore)
	}
	if idfChore >= defaultThreshold {
		t.Errorf("two unrelated chores still score %.4f, at or above the default threshold %.2f", idfChore, defaultThreshold)
	}
	if idfTmpl >= defaultThreshold {
		t.Errorf("two template siblings still score %.4f, at or above the default threshold %.2f", idfTmpl, defaultThreshold)
	}
}

// TestClusterDuplicatesUsesCompleteLinkage pins the property that makes grouping
// safe. Single linkage — merging any two issues joined by a PATH of pairwise
// similarities — collapsed 898 of 2,693 issues into one cluster when measured on
// a real backlog, which flattered the ranking for entirely the wrong reason.
func TestClusterDuplicatesUsesCompleteLinkage(t *testing.T) {
	ab := &types.Issue{ID: "a"}
	bb := &types.Issue{ID: "b"}
	cb := &types.Issue{ID: "c"}
	// a~b and b~c are reported, a~c is NOT. Single linkage would make {a,b,c}
	// one cluster; complete linkage must not.
	pairs := []duplicatePair{
		{IssueA: ab, IssueB: bb, Similarity: 0.9},
		{IssueA: bb, IssueB: cb, Similarity: 0.8},
	}
	clusters := clusterDuplicates(pairs)
	for _, c := range clusters {
		if len(c.Issues) > 2 {
			t.Fatalf("chained a cluster of %d through an unreported pair; complete linkage must not merge a~b~c when a~c is absent", len(c.Issues))
		}
	}
	if len(clusters) != 1 {
		t.Errorf("got %d clusters, want 1 (a~b claims both; c has no unclaimed partner)", len(clusters))
	}
}

// TestClusterDuplicatesCollapsesAFamily is the other half: a set of mutually
// similar issues must become ONE finding, not one per pair.
func TestClusterDuplicatesCollapsesAFamily(t *testing.T) {
	var issues []*types.Issue
	for i := 0; i < 5; i++ {
		issues = append(issues, &types.Issue{ID: fmt.Sprintf("f%d", i)})
	}
	var pairs []duplicatePair
	for i := range issues {
		for j := i + 1; j < len(issues); j++ {
			pairs = append(pairs, duplicatePair{IssueA: issues[i], IssueB: issues[j], Similarity: 0.9})
		}
	}
	if len(pairs) != 10 {
		t.Fatalf("fixture built %d pairs, want 10", len(pairs))
	}
	clusters := clusterDuplicates(pairs)
	if len(clusters) != 1 {
		t.Fatalf("a mutually similar family of 5 produced %d clusters, want 1", len(clusters))
	}
	if len(clusters[0].Issues) != 5 {
		t.Errorf("cluster holds %d issues, want all 5", len(clusters[0].Issues))
	}
}

// TestWeightedSimilarityReducesToUnweighted pins the compatibility property the
// --weighting raw escape hatch depends on.
func TestWeightedSimilarityReducesToUnweighted(t *testing.T) {
	a := map[string]int{"fix": 2, "auth": 1, "bug": 1}
	b := map[string]int{"fix": 1, "auth": 3, "login": 1}

	for name, weights := range map[string]tokenWeights{
		"nil":  nil,
		"unit": {"fix": 1, "auth": 1, "bug": 1, "login": 1},
	} {
		if got, want := weightedJaccardSimilarity(a, b, weights), jaccardSimilarity(a, b); got != want {
			t.Errorf("weighted Jaccard with %s weights = %v, want %v", name, got, want)
		}
		if got, want := weightedCosineSimilarity(a, b, weights), cosineSimilarity(a, b); got != want {
			t.Errorf("weighted cosine with %s weights = %v, want %v", name, got, want)
		}
	}
}

// TestInverseDocumentFrequenciesZeroesUbiquitousTokens pins the mechanism.
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
	if s := pairSimilarity(docs[0], docs[1], weights); s != 0 {
		t.Errorf("pair sharing only a ubiquitous token scored %v, want 0", s)
	}
}

// TestFieldSimilarityFallsBackWhenADescriptionIsMissing: an absent description
// must not scale down a good title match.
func TestFieldSimilarityFallsBackWhenADescriptionIsMissing(t *testing.T) {
	a := issueFields{title: map[string]int{"airplay": 1, "android": 1}}
	b := issueFields{title: map[string]int{"airplay": 1, "android": 1}}
	idf := tokenWeights{"airplay": 1, "android": 1}

	if got := fieldSimilarity(a, b, idf, nil); got < 1-1e-9 {
		t.Errorf("identical titles with no descriptions scored %v, want 1 (no dilution by a missing field)", got)
	}
	// With descriptions present and disjoint, the score must drop below 1 but
	// stay dominated by the title.
	a.desc = map[string]int{"alpha": 1}
	b.desc = map[string]int{"beta": 1}
	descIDF := tokenWeights{"alpha": 1, "beta": 1}
	got := fieldSimilarity(a, b, idf, descIDF)
	if got >= 1 || got < titleFieldWeight-0.001 {
		t.Errorf("identical titles with disjoint descriptions scored %v, want in [%v, 1)", got, titleFieldWeight)
	}
}
