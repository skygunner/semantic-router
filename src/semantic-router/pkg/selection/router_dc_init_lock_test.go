package selection

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRouterDCEmbeddingBuildDoesNotBlockReaders(t *testing.T) {
	r := NewRouterDCSelector(DefaultRouterDCConfig())
	r.SetModelEmbedding("existing", []float32{1})
	entered, release := make(chan struct{}), make(chan struct{})
	r.setContextEmbeddingFunc(func(context.Context, string) ([]float32, error) {
		close(entered)
		<-release
		return []float32{2}, nil
	})
	done := make(chan error, 1)
	go func() {
		done <- r.InitializeFromConfig(map[string]config.ModelParams{"new": {Description: "new model"}})
	}()
	<-entered
	read := make(chan map[string][]float32, 1)
	go func() { read <- r.GetModelEmbeddings() }()
	select {
	case models := <-read:
		if len(models) != 1 || models["existing"][0] != 1 {
			t.Errorf("read during build = %v, want existing embeddings", models)
		}
	case <-time.After(time.Second):
		t.Error("embedding build held the readers' lock during an external call")
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
