# Future direction: GNN classification on MNIST

This directory preserves the original investigation of treating image pixels
as graph nodes and applying GCN/GAT-style layers to MNIST classification.

It is a separate modeling direction from the supported inference compiler. The
legacy notebook and source are incomplete, are not installed as part of the
package, and are not used to support current performance claims. They remain
here so the research can be resumed without confusing exploratory model code
with the tested systems implementation.

A proper continuation should start with:

1. a clean message-passing baseline using a maintained graph library;
2. registered `ModuleList` layers and correct class-dimension normalization;
3. train/validation/test separation with repeatable seeds;
4. comparison against a parameter-matched CNN;
5. profiling of adjacency construction, message passing, and memory;
6. experiments with relation-typed grid edges that connect to the structured
   operator in `docs/DESIGN.md`.
