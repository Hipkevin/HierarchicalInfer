# Can Large Language Models Accurately Discriminate Subject Term Hierarchical Relationship?

![framework.png](framework.png "Framework for Discriminating Subject Term Hierarchical Relationships")

## Subject Terms Recall

**R.S1 Co-occurrence**: count the co-occurrence relationships in author keywords, and when the frequency of keyword co-occurrence pairs is greater than the retrieval year interval (14 years), the two keywords in the keyword co-occurrence pairs are used as candidate subject term pairs.

****R.S2 Co-occurrence Cluster**: construct the co-occurrence frequency matrix of author keywords on the basis of R.S1, use this matrix to perform K-Means clustering, select the K value corresponding to the inflection point of the SSE curve as the number of clusters according to the principle of the elbow method, and then arrange and combine the keywords in each cluster according to the Cn2 permutation and take them as candidate subject term pairs.

**R.S3 LLMs Embedding Cluster**: also based on R.S1, the embedding vectors of the author's keywords are obtained by using a language model with a smaller number of parameters after distillation, and the candidate subject term pairs are obtained after the LLM characterization is recalled according to the clustering and permutation methods in R.S2.

## Hierarchical Relationship Discrimination

**D.S1 Subsumption Rule**: For keyword x and keyword y of a candidate subject term pair, P(x|y) and P(y|x) are computed, and x is the superlative of y when P(x|y)=1 and P(y|x)<1. Usually, the condition P(x|y)=1 is relaxed to P(x|y)>alpha, and alpha is chosen according to different domains and data sizes, usually 0.8.

**D.S2 Klink**: semantic features are introduced on the basis of D.S1 to compute L(x,y)=(P(x|y) - P(y|x)) * c(x,y) * (1 + N(x,y)). c(x,y) denotes the cosine similarity of keywords x and y in the co-occurrence matrix, and N(x,y) denotes the string similarity of keywords x and y. In this study, the longest common subsequence distance (LCS) is used. x is the superordinate of y for L(x,y)>t, and t is usually taken as 0.2.

**D.S3 LLMs Prompt Engineering**: the hierarchical relationship of each candidate subject term pair is discriminated by Prompt Engineering. The Prompt template designed in this study is as follows:

`Hypernymy and hyponymy are the semantic relations between a generic term (hypernym) and a more specific term (hyponym). Determine the hierarchical relationship between two words based on subject classification. Answer 1 if {w1} is the superordinate of {w2}, 0 if {w2} is the superordinate of {w1}, or -1 if there is no superordinate relationship between the two.Do not output any text other than 1, 0, and -1.`