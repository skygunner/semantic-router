/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// sharedPrefix is longer than the 32-byte window the old hashQuery truncated to.
const sharedPrefix = "You are a helpful assistant. Answer the user request: "

func TestRouterDCSelector_HashQueryDistinguishesSharedPrefix(t *testing.T) {
	selector := NewRouterDCSelector(DefaultRouterDCConfig())

	poem := sharedPrefix + "write a poem about the ocean"
	sql := sharedPrefix + "SQL to drop the production users table"

	poemHash := selector.hashQuery(poem)
	sqlHash := selector.hashQuery(sql)

	if poemHash == sqlHash {
		t.Fatalf("queries sharing a %d-byte prefix hashed to the same key %q", len(sharedPrefix), poemHash)
	}
	if again := selector.hashQuery(poem); again != poemHash {
		t.Fatalf("hashQuery is not stable: %q != %q", again, poemHash)
	}
	if len(poemHash) != 64 || strings.Trim(poemHash, "0123456789abcdef") != "" {
		t.Fatalf("expected 64-char lowercase hex SHA-256 digest, got %q", poemHash)
	}
}

func TestRouterDCSelector_UpdateFeedbackKeysAffinityPerQuery(t *testing.T) {
	selector := NewRouterDCSelector(DefaultRouterDCConfig())
	ctx := context.Background()

	poem := sharedPrefix + "write a poem about the ocean"
	sql := sharedPrefix + "SQL to drop the production users table"

	if err := selector.UpdateFeedback(ctx, &Feedback{Query: poem, WinnerModel: "poet"}); err != nil {
		t.Fatalf("UpdateFeedback(poem) error = %v", err)
	}
	if err := selector.UpdateFeedback(ctx, &Feedback{Query: sql, WinnerModel: "dba"}); err != nil {
		t.Fatalf("UpdateFeedback(sql) error = %v", err)
	}

	selector.affinityMu.RLock()
	defer selector.affinityMu.RUnlock()

	if got := len(selector.affinityMatrix); got != 2 {
		t.Fatalf("expected 2 affinity buckets, got %d: %v", got, selector.affinityMatrix)
	}

	poemBucket := selector.affinityMatrix[selector.hashQuery(poem)]
	sqlBucket := selector.affinityMatrix[selector.hashQuery(sql)]

	if poemBucket["poet"] == 0 || poemBucket["dba"] != 0 {
		t.Fatalf("poem bucket should only hold poet affinity, got %v", poemBucket)
	}
	if sqlBucket["dba"] == 0 || sqlBucket["poet"] != 0 {
		t.Fatalf("sql bucket should only hold dba affinity, got %v", sqlBucket)
	}
}

func TestRouterDCSelector_InitializeFromConfigStopsWhenEmbeddingModelIsNotPrepared(t *testing.T) {
	selector := NewRouterDCSelector(DefaultRouterDCConfig())
	calls := 0
	selector.setContextEmbeddingFunc(func(context.Context, string) ([]float32, error) {
		calls++
		_, err := embedding.NewSet(nil, "mmbert").Get("mmbert", 0, 0)
		return nil, err
	})

	err := selector.InitializeFromConfig(map[string]config.ModelParams{
		"a": {Description: "fast"},
		"b": {Description: "careful"},
		"c": {Description: "cheap"},
	})
	if err != nil {
		t.Fatalf("InitializeFromConfig() error = %v", err)
	}
	if calls != 1 {
		t.Fatalf("embedding calls = %d, want 1: an unprepared model fails for every description", calls)
	}
	if len(selector.modelEmbeddings) != 0 {
		t.Fatalf("model embeddings = %v, want none", selector.modelEmbeddings)
	}
}

func TestRouterDCInitializationBuildsOutsideLockAndCancelsAtomically(t *testing.T) {
	r := NewRouterDCSelector(DefaultRouterDCConfig())
	r.SetModelEmbedding("existing", []float32{1})
	entered := make(chan struct{})
	release := make(chan struct{})
	r.setContextEmbeddingFunc(func(ctx context.Context, text string) ([]float32, error) {
		close(entered)
		select {
		case <-release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return []float32{2}, nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- r.InitializeFromConfigContext(ctx, map[string]config.ModelParams{"new": {Description: "new model"}})
	}()
	<-entered
	read := make(chan map[string][]float32, 1)
	go func() { read <- r.GetModelEmbeddings() }()
	select {
	case models := <-read:
		if len(models) != 1 || models["existing"][0] != 1 {
			t.Fatalf("observed partial state: %v", models)
		}
	case <-time.After(time.Second):
		t.Fatal("state read blocked by embedding build")
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
	if _, ok := r.GetModelEmbeddings()["new"]; ok {
		t.Fatal("cancelled build published embedding")
	}
	close(release)
}

func TestFactoryCreateAllPassesInitContextToRouterDC(t *testing.T) {
	initCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var gotCtx context.Context
	registry := NewFactory(&ModelSelectionConfig{Method: string(MethodRouterDC)}).
		WithModelConfig(map[string]config.ModelParams{"model": {Description: "fast"}}).
		WithContextEmbeddingFunc(func(ctx context.Context, _ string, _ EmbeddingConfig) ([]float32, error) {
			gotCtx = ctx
			return []float32{1}, nil
		}, EmbeddingConfig{}).
		WithInitContext(initCtx).
		CreateAll()
	selector, ok := registry.Get(MethodRouterDC)
	if !ok {
		t.Fatal("RouterDC selector missing from the created registry")
	}
	dc, ok := selector.(*RouterDCSelector)
	if !ok {
		t.Fatalf("registry selector type = %T", selector)
	}
	if len(dc.GetModelEmbeddings()) != 1 {
		t.Fatalf("published embeddings = %d, want 1", len(dc.GetModelEmbeddings()))
	}
	if gotCtx != initCtx {
		t.Fatal("the embedding func did not receive the factory init context")
	}
}

func TestFactoryInitContextCancellationLeavesNoPartialEmbeddings(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})
	type registryResult struct {
		registry *Registry
	}
	done := make(chan registryResult, 1)
	go func() {
		done <- registryResult{NewFactory(&ModelSelectionConfig{Method: string(MethodRouterDC)}).
			WithModelConfig(map[string]config.ModelParams{"model": {Description: "slow"}}).
			WithContextEmbeddingFunc(func(ctx context.Context, _ string, _ EmbeddingConfig) ([]float32, error) {
				close(started)
				<-ctx.Done()
				return nil, ctx.Err()
			}, EmbeddingConfig{}).
			WithInitContext(ctx).
			CreateAll()}
	}()
	<-started
	cancel()
	select {
	case result := <-done:
		selector, ok := result.registry.Get(MethodRouterDC)
		if !ok {
			t.Fatal("RouterDC selector missing from the created registry")
		}
		dc, ok := selector.(*RouterDCSelector)
		if !ok {
			t.Fatalf("registry selector type = %T", selector)
		}
		if embeddings := dc.GetModelEmbeddings(); len(embeddings) != 0 {
			t.Fatalf("cancelled initialization published %d embeddings, want 0", len(embeddings))
		}
	case <-time.After(5 * time.Second):
		t.Fatal("CreateAll did not return after the init context was cancelled")
	}
}
