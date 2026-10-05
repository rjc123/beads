package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"unicode"

	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/spf13/cobra"
	"github.com/steveyegge/beads/internal/config"
	"github.com/steveyegge/beads/internal/metrics"
	"github.com/steveyegge/beads/internal/telemetry"
	"github.com/steveyegge/beads/internal/types"
	"github.com/steveyegge/beads/internal/ui"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

var findDuplicatesCmd = &cobra.Command{
	Use:     "find-duplicates",
	Aliases: []string{"find-dups"},
	GroupID: "views",
	Short:   "Find semantically similar issues using text analysis or AI",
	Long: `Find issues that are semantically similar but not exact duplicates.

Unlike 'bd duplicates' which finds exact content matches, find-duplicates
uses text similarity or AI to find issues that discuss the same topic
with different wording.

Approaches:
  mechanical  Token-based text similarity (default, no API key needed)
  ai          LLM-based semantic comparison (requires ANTHROPIC_API_KEY, MINIMAX_API_KEY, or ai.api_key)

The mechanical approach tokenizes titles and descriptions, weights each
token by its inverse document frequency across the corpus, then blends
Jaccard and cosine similarity. IDF weighting is what makes the ranking usable
on a large backlog: the words almost every issue carries (add, fix, update,
and whatever description template the tracker uses) weigh nearly nothing, so
a score is driven by the rare words two issues share. Title tokens count for
more than description tokens. It's fast and free but may miss semantically
similar issues with very different wording.

--weighting raw restores the unweighted scoring used before IDF, in which a
pair's score was dominated by common-token overlap.

The AI approach sends candidate pairs to an Anthropic-compatible model for semantic comparison.
It first uses mechanical pre-filtering to reduce the number of API calls,
then asks the LLM to judge whether the remaining pairs are true duplicates.

Examples:
  bd find-duplicates                       # Mechanical similarity (default)
  bd find-duplicates --threshold 0.4       # Lower threshold = more results
  bd find-duplicates --method ai           # Use AI for semantic comparison
  bd find-duplicates --status open         # Only check open issues
  bd find-duplicates --limit 20            # Show top 20 pairs
  bd find-duplicates --json                # JSON output
  bd find-duplicates --weighting raw       # Unweighted (pre-IDF) scoring`,
	SilenceUsage:  true,
	SilenceErrors: true,
	RunE:          runFindDuplicates,
}

func init() {
	findDuplicatesCmd.Flags().String("method", "mechanical", "Detection method: mechanical, ai")
	findDuplicatesCmd.Flags().String("weighting", weightingIDF, "Token weighting for mechanical similarity: idf, raw")
	findDuplicatesCmd.Flags().Float64("threshold", 0.5, "Similarity threshold (0.0-1.0, lower = more results)")
	findDuplicatesCmd.Flags().StringP("status", "s", "", "Filter by status (default: non-closed)")
	findDuplicatesCmd.Flags().IntP("limit", "n", 50, "Maximum number of pairs to show")
	findDuplicatesCmd.Flags().String("model", "", "AI model to use (only with --method ai; default from config ai.model)")
	// Defensive row cap (be-x42v): exits 2 on overage, default disabled.
	addMaxRowsFlag(findDuplicatesCmd)
	rootCmd.AddCommand(findDuplicatesCmd)
}

// Token weightings available to the mechanical method.
const (
	// weightingIDF weights each token by log(N/df) across the corpus.
	weightingIDF = "idf"
	// weightingRaw weights every token equally, as beads did before IDF.
	weightingRaw = "raw"
)

// duplicatePair represents a pair of potentially duplicate issues.
type duplicatePair struct {
	IssueA     *types.Issue `json:"issue_a"`
	IssueB     *types.Issue `json:"issue_b"`
	Similarity float64      `json:"similarity"`
	Method     string       `json:"method"`
	Reason     string       `json:"reason,omitempty"`
}

func runFindDuplicates(cmd *cobra.Command, _ []string) error {
	evt := metrics.NewCommandEvent("find-duplicates")
	defer func() {
		if c := metrics.Global(); c != nil {
			c.CloseEventAndAdd(evt)
		}
	}()

	method, _ := cmd.Flags().GetString("method")
	weighting, _ := cmd.Flags().GetString("weighting")
	threshold, _ := cmd.Flags().GetFloat64("threshold")
	status, _ := cmd.Flags().GetString("status")
	limit, _ := cmd.Flags().GetInt("limit")
	model, _ := cmd.Flags().GetString("model")
	if model == "" {
		_, keySource := config.ResolveAIAPIKey("")
		model = config.DefaultAIModelFor(keySource)
	}

	if method != "mechanical" && method != "ai" {
		return HandleErrorRespectJSON("invalid method %q (use: mechanical, ai)", method)
	}

	if weighting != weightingIDF && weighting != weightingRaw {
		return HandleErrorRespectJSON("invalid weighting %q (use: idf, raw)", weighting)
	}

	if method == "ai" {
		if apiKey, _ := config.ResolveAIAPIKey(""); apiKey == "" {
			return HandleErrorRespectJSON("--method ai requires ANTHROPIC_API_KEY, MINIMAX_API_KEY, or ai.api_key in config")
		}
	}

	// Fetch issues
	maxRows, maxRowsSource, err := resolveMaxRows(cmd)
	if err != nil {
		return err
	}
	filter := types.IssueFilter{
		MaxRows:       maxRows,
		MaxRowsSource: maxRowsSource,
	}
	if status != "" && status != "all" {
		s := types.Status(status)
		filter.Status = &s
	}

	if usesProxiedServer() {
		// maxRows was already resolved above (to build filter); reject using
		// that value directly instead of re-resolving via
		// rejectMaxRowsUnderProxiedServer, which would call resolveMaxRows a
		// second time and double any malformed-env warning it emits.
		if err := rejectResolvedMaxRowsUnderProxiedServer(maxRows); err != nil {
			return err
		}
		return runFindDuplicatesProxiedServer(rootCtx, filter, status, method, weighting, threshold, limit, model)
	}

	issues, err := store.SearchIssues(rootCtx, "", filter)
	if err != nil {
		if capErr := handleMaxRowsError(err); capErr != nil {
			return capErr
		}
		return HandleErrorRespectJSON("fetching issues: %v", err)
	}
	issues = filterClosedIfNoStatus(issues, status)

	return reportFindDuplicates(rootCtx, issues, method, weighting, threshold, limit, model)
}

func filterClosedIfNoStatus(issues []*types.Issue, status string) []*types.Issue {
	if status != "" {
		return issues
	}
	var filtered []*types.Issue
	for _, issue := range issues {
		if issue.Status != types.StatusClosed {
			filtered = append(filtered, issue)
		}
	}
	return filtered
}

func reportFindDuplicates(ctx context.Context, issues []*types.Issue, method, weighting string, threshold float64, limit int, model string) error {
	if len(issues) < 2 {
		if jsonOutput {
			return outputJSON(map[string]interface{}{
				"pairs": []interface{}{},
				"count": 0,
			})
		}
		fmt.Println("Not enough issues to compare (need at least 2)")
		return nil
	}

	// Find duplicate pairs
	var pairs []duplicatePair
	switch method {
	case "mechanical":
		pairs = findMechanicalDuplicatesWeighted(issues, threshold, weighting)
	case "ai":
		pairs = findAIDuplicates(ctx, issues, threshold, weighting, model)
	}

	// Sort by similarity (highest first). Stable, so that pairs scoring
	// equally stay in the deterministic (issue A, issue B) order they were
	// generated in rather than permuting between runs.
	sort.SliceStable(pairs, func(i, j int) bool {
		return pairs[i].Similarity > pairs[j].Similarity
	})

	// Apply limit
	if limit > 0 && len(pairs) > limit {
		pairs = pairs[:limit]
	}

	// Output
	if jsonOutput {
		type pairJSON struct {
			IssueAID    string  `json:"issue_a_id"`
			IssueBID    string  `json:"issue_b_id"`
			IssueATitle string  `json:"issue_a_title"`
			IssueBTitle string  `json:"issue_b_title"`
			Similarity  float64 `json:"similarity"`
			Method      string  `json:"method"`
			Reason      string  `json:"reason,omitempty"`
		}
		jsonPairs := make([]pairJSON, len(pairs))
		for i, p := range pairs {
			jsonPairs[i] = pairJSON{
				IssueAID:    p.IssueA.ID,
				IssueBID:    p.IssueB.ID,
				IssueATitle: p.IssueA.Title,
				IssueBTitle: p.IssueB.Title,
				Similarity:  p.Similarity,
				Method:      p.Method,
				Reason:      p.Reason,
			}
		}
		return outputJSON(map[string]interface{}{
			"pairs":     jsonPairs,
			"count":     len(jsonPairs),
			"method":    method,
			"weighting": weighting,
			"threshold": threshold,
		})
	}

	if len(pairs) == 0 {
		fmt.Printf("No similar issues found (threshold: %.0f%%)\n", threshold*100)
		return nil
	}

	fmt.Printf("%s Found %d potential duplicate pair(s) (threshold: %.0f%%):\n\n",
		ui.RenderWarn("🔍"), len(pairs), threshold*100)

	for i, p := range pairs {
		pct := p.Similarity * 100
		fmt.Printf("%s Pair %d (%.0f%% similar):\n", ui.RenderAccent("━━"), i+1, pct)
		fmt.Printf("  %s %s\n", ui.RenderPass(p.IssueA.ID), p.IssueA.Title)
		fmt.Printf("  %s %s\n", ui.RenderPass(p.IssueB.ID), p.IssueB.Title)
		if p.Reason != "" {
			fmt.Printf("  %s %s\n", ui.RenderAccent("Reason:"), p.Reason)
		}
		fmt.Printf("  %s bd show %s %s\n\n", ui.RenderAccent("Compare:"), p.IssueA.ID, p.IssueB.ID)
	}
	return nil
}

// tokenize splits text into lowercase word tokens, removing punctuation.
func tokenize(text string) map[string]int {
	tokens := make(map[string]int)
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-'
	})
	for _, w := range words {
		if len(w) > 1 { // Skip single chars
			tokens[w]++
		}
	}
	return tokens
}

// issueText returns the combined text content of an issue for comparison.
func issueText(issue *types.Issue) string {
	parts := []string{issue.Title}
	if issue.Description != "" {
		parts = append(parts, issue.Description)
	}
	return strings.Join(parts, " ")
}

// tokenWeights assigns each token the contribution it makes to a similarity
// score. A nil map weighs every token 1.0, which reproduces the unweighted
// scoring exactly.
type tokenWeights map[string]float64

func (w tokenWeights) weight(token string) float64 {
	if w == nil {
		return 1
	}
	return w[token]
}

// documentFrequencies counts how many documents contain each token.
func documentFrequencies(docs []map[string]int) map[string]int {
	df := make(map[string]int)
	for _, doc := range docs {
		for token := range doc {
			df[token]++
		}
	}
	return df
}

// inverseDocumentFrequencies turns document frequencies into IDF weights,
// log(N/df). A token present in every document weighs exactly zero, so the
// boilerplate that every issue shares contributes nothing to a score, and a
// token present in a handful of documents dominates.
func inverseDocumentFrequencies(docs []map[string]int) tokenWeights {
	df := documentFrequencies(docs)
	n := len(docs)
	weights := make(tokenWeights, len(df))
	for token, count := range df {
		if count <= 0 || count > n {
			continue
		}
		weights[token] = math.Log(float64(n) / float64(count))
	}
	return weights
}

// jaccardSimilarity computes the Jaccard similarity between two token sets.
func jaccardSimilarity(a, b map[string]int) float64 {
	return weightedJaccardSimilarity(a, b, nil)
}

// weightedJaccardSimilarity computes the Jaccard similarity between two token
// multisets, with each token contributing its weight in place of 1.
func weightedJaccardSimilarity(a, b map[string]int, weights tokenWeights) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 0
	}

	intersection := 0.0
	union := 0.0

	// Count union from a
	for token, countA := range a {
		w := weights.weight(token)
		if w == 0 {
			// A zero-weight token cannot change either total.
			continue
		}
		if countB, ok := b[token]; ok {
			intersection += w * float64(min(countA, countB))
			union += w * float64(max(countA, countB))
		} else {
			union += w * float64(countA)
		}
	}
	// Count tokens only in b
	for token, countB := range b {
		if _, ok := a[token]; ok {
			continue
		}
		union += weights.weight(token) * float64(countB)
	}

	if union == 0 {
		return 0
	}
	return intersection / union
}

// cosineSimilarity computes the cosine similarity between two token vectors.
func cosineSimilarity(a, b map[string]int) float64 {
	return weightedCosineSimilarity(a, b, nil)
}

// weightedCosineSimilarity computes the cosine similarity between two token
// vectors, with each token count scaled by that token's weight.
func weightedCosineSimilarity(a, b map[string]int, weights tokenWeights) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	dotProduct := 0.0
	magA := 0.0
	magB := 0.0

	for token, countA := range a {
		w := weights.weight(token)
		va := w * float64(countA)
		magA += va * va
		if countB, ok := b[token]; ok {
			dotProduct += va * w * float64(countB)
		}
	}
	for token, countB := range b {
		vb := weights.weight(token) * float64(countB)
		magB += vb * vb
	}

	if magA == 0 || magB == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(magA) * math.Sqrt(magB))
}

// pairSimilarity is the score reported for one candidate pair: the average of
// Jaccard and cosine, which disagree in useful ways about short text.
func pairSimilarity(a, b map[string]int, weights tokenWeights) float64 {
	return (weightedJaccardSimilarity(a, b, weights) + weightedCosineSimilarity(a, b, weights)) / 2
}

// titleTokenWeight is how many times a title token counts relative to a
// description token. An issue's title is the closest thing it has to an
// identity, while a description often carries a template every issue in the
// tracker shares. Measured on a 2,693-issue backlog, lifting the title from
// 1 to 3 moved a known true duplicate pair from rank 5,461 to rank 1,454 and
// pushed a known false pair from rank 20,232 to rank 222,294.
const titleTokenWeight = 3

// issueTokens tokenizes an issue for comparison, counting each title token
// titleTokenWeight times.
func issueTokens(issue *types.Issue) map[string]int {
	tokens := make(map[string]int)
	for token, count := range tokenize(issue.Title) {
		tokens[token] = count * titleTokenWeight
	}
	for token, count := range tokenize(issue.Description) {
		tokens[token] += count
	}
	return tokens
}

// findMechanicalDuplicates finds similar issues using token-based text
// similarity weighted by inverse document frequency.
func findMechanicalDuplicates(issues []*types.Issue, threshold float64) []duplicatePair {
	return findMechanicalDuplicatesWeighted(issues, threshold, weightingIDF)
}

// findMechanicalDuplicatesWeighted scores every issue pair under the named
// token weighting. weightingRaw reproduces the pre-IDF behaviour exactly:
// unit token weights over title and description read as one flat document.
func findMechanicalDuplicatesWeighted(issues []*types.Issue, threshold float64, weighting string) []duplicatePair {
	unweighted := weighting == weightingRaw

	// Pre-tokenize all issues
	docs := make([]map[string]int, len(issues))
	for i, issue := range issues {
		if unweighted {
			docs[i] = tokenize(issueText(issue))
		} else {
			docs[i] = issueTokens(issue)
		}
	}

	var weights tokenWeights
	if !unweighted {
		weights = inverseDocumentFrequencies(docs)
	}

	var pairs []duplicatePair

	// Compare all pairs
	for i := 0; i < len(docs); i++ {
		for j := i + 1; j < len(docs); j++ {
			similarity := pairSimilarity(docs[i], docs[j], weights)
			if similarity >= threshold {
				pairs = append(pairs, duplicatePair{
					IssueA:     issues[i],
					IssueB:     issues[j],
					Similarity: similarity,
					Method:     "mechanical",
				})
			}
		}
	}

	return pairs
}

// findAIDuplicates uses LLM-based semantic comparison to find duplicates.
// It first pre-filters with mechanical similarity to reduce API calls.
func findAIDuplicates(ctx context.Context, issues []*types.Issue, threshold float64, weighting, model string) []duplicatePair {
	// Pre-filter: use mechanical similarity with a lower threshold to find candidates
	preFilterThreshold := threshold * 0.5 // Cast a wider net for pre-filtering
	if preFilterThreshold < 0.15 {
		preFilterThreshold = 0.15
	}
	candidates := findMechanicalDuplicatesWeighted(issues, preFilterThreshold, weighting)

	if len(candidates) == 0 {
		return nil
	}

	// Limit candidates to avoid excessive API calls
	maxCandidates := 100
	if len(candidates) > maxCandidates {
		// Sort by mechanical similarity and take the top candidates
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Similarity > candidates[j].Similarity
		})
		candidates = candidates[:maxCandidates]
	}

	fmt.Fprintf(os.Stderr, "Analyzing %d candidate pairs with AI...\n", len(candidates))

	apiKey, keySource := config.ResolveAIAPIKey("")
	clientOptions := []option.RequestOption{option.WithAPIKey(apiKey)}
	if baseURL := config.DefaultAIBaseURL(keySource); baseURL != "" {
		clientOptions = append(clientOptions, option.WithBaseURL(baseURL))
	}
	client := anthropic.NewClient(clientOptions...)

	var pairs []duplicatePair

	// Batch candidates into groups for efficient API usage
	batchSize := 10
	for i := 0; i < len(candidates); i += batchSize {
		end := i + batchSize
		if end > len(candidates) {
			end = len(candidates)
		}
		batch := candidates[i:end]

		results := analyzeWithAI(ctx, client, model, batch)
		for _, r := range results {
			if r.Similarity >= threshold {
				pairs = append(pairs, r)
			}
		}
	}

	return pairs
}

// analyzeWithAI sends a batch of candidate pairs to the LLM for semantic comparison.
func analyzeWithAI(ctx context.Context, client anthropic.Client, model anthropic.Model, candidates []duplicatePair) []duplicatePair {
	if len(candidates) == 0 {
		return nil
	}

	// Build the prompt
	var sb strings.Builder
	sb.WriteString("You are analyzing issue pairs to determine if they are semantic duplicates.\n")
	sb.WriteString("For each pair, determine if they describe the same problem/task/feature.\n")
	sb.WriteString("Respond with a JSON array of objects, one per pair, with fields:\n")
	sb.WriteString("  - pair_index (int): 0-based index of the pair\n")
	sb.WriteString("  - is_duplicate (bool): true if semantically the same issue\n")
	sb.WriteString("  - confidence (float): 0.0-1.0 how confident you are\n")
	sb.WriteString("  - reason (string): brief explanation\n\n")
	sb.WriteString("Respond ONLY with the JSON array, no other text.\n\n")

	for i, c := range candidates {
		fmt.Fprintf(&sb, "--- Pair %d ---\n", i)
		fmt.Fprintf(&sb, "Issue A [%s]: %s\n", c.IssueA.ID, c.IssueA.Title)
		if c.IssueA.Description != "" {
			descA := c.IssueA.Description
			if len(descA) > 500 {
				descA = descA[:500] + "..."
			}
			fmt.Fprintf(&sb, "  Description: %s\n", descA)
		}
		fmt.Fprintf(&sb, "Issue B [%s]: %s\n", c.IssueB.ID, c.IssueB.Title)
		if c.IssueB.Description != "" {
			descB := c.IssueB.Description
			if len(descB) > 500 {
				descB = descB[:500] + "..."
			}
			fmt.Fprintf(&sb, "  Description: %s\n", descB)
		}
		sb.WriteString("\n")
	}

	tracer := telemetry.Tracer("github.com/steveyegge/beads/ai")
	aiCtx, aiSpan := tracer.Start(ctx, "anthropic.messages.new")
	aiSpan.SetAttributes(
		attribute.String("bd.ai.model", model),
		attribute.String("bd.ai.operation", "find_duplicates"),
		attribute.Int("bd.ai.batch_size", len(candidates)),
	)
	t0 := time.Now()
	message, err := client.Messages.New(aiCtx, anthropic.MessageNewParams{
		Model:     model,
		MaxTokens: 2048,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(sb.String())),
		},
	})
	if err != nil {
		aiSpan.RecordError(err)
		aiSpan.SetStatus(codes.Error, err.Error())
		aiSpan.End()
		fmt.Fprintf(os.Stderr, "Warning: AI analysis failed: %v\n", err)
		// Fall back to mechanical scores
		return candidates
	}
	aiSpan.SetAttributes(
		attribute.Int64("bd.ai.input_tokens", message.Usage.InputTokens),
		attribute.Int64("bd.ai.output_tokens", message.Usage.OutputTokens),
		attribute.Float64("bd.ai.duration_ms", float64(time.Since(t0).Milliseconds())),
	)
	aiSpan.End()

	if len(message.Content) == 0 || message.Content[0].Type != "text" {
		fmt.Fprintf(os.Stderr, "Warning: unexpected AI response format\n")
		return candidates
	}

	// Parse the JSON response
	responseText := message.Content[0].Text

	// Try to extract JSON from the response (handle markdown code blocks)
	jsonText := responseText
	if idx := strings.Index(jsonText, "["); idx >= 0 {
		jsonText = jsonText[idx:]
	}
	if idx := strings.LastIndex(jsonText, "]"); idx >= 0 {
		jsonText = jsonText[:idx+1]
	}

	var results []struct {
		PairIndex   int     `json:"pair_index"`
		IsDuplicate bool    `json:"is_duplicate"`
		Confidence  float64 `json:"confidence"`
		Reason      string  `json:"reason"`
	}

	if err := json.Unmarshal([]byte(jsonText), &results); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to parse AI response: %v\n", err)
		return candidates
	}

	// Map results back to candidate pairs
	var pairs []duplicatePair
	for _, r := range results {
		if r.PairIndex < 0 || r.PairIndex >= len(candidates) {
			continue
		}
		if r.IsDuplicate {
			c := candidates[r.PairIndex]
			pairs = append(pairs, duplicatePair{
				IssueA:     c.IssueA,
				IssueB:     c.IssueB,
				Similarity: r.Confidence,
				Method:     "ai",
				Reason:     r.Reason,
			})
		}
	}

	return pairs
}
