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

## Experiments
Updated 03-13-25

<div style="text-align: center;">
<table style="margin: auto; text-align: center;">
<tr style="font-weight: bold">
<td rowspan="2" colspan="1">Phase</td>
<td rowspan="2" colspan="2">Strategy</td>
<td rowspan="1" colspan="3">OpenAlex</td>
<td rowspan="1" colspan="3">CSO</td>
</tr>
<tr style="font-weight: bold">
<td>P</td>
<td>R</td>
<td>F1</td>
<td>P</td>
<td>R</td>
<td>F1</td>
</tr>
<tr>
<td rowspan="3" colspan="1">Recall</td>
<td>R.S1</td>
<td>screen</td>
<td>33.51%</td>
<td>1.05%</td>
<td>2.04%</td>
<td>33.22%</td>
<td>2.83%</td>
<td>5.22% </td>
</tr>
<tr>
<td>R.S2</td>
<td>qw32b-cluster</td>
<td>4.61%</td>
<td>0.08%</td>
<td>0.16%</td>
<td>3.19%</td>
<td>0.15%</td>
<td>0.29% </td>
</tr>
<tr>
<td>R.S3</td>
<td>co-cluster</td>
<td>2.58%</td>
<td>0.05%</td>
<td>0.09%</td>
<td>2.29%</td>
<td>0.12%</td>
<td>0.22% </td>
</tr>
<tr>
<td rowspan="12" colspan="1">Discrimination</td>
<td>D.S1</td>
<td>Subsumption</td>
<td>4.05%</td>
<td>2.36%</td>
<td>2.99%</td>
<td>3.45%</td>
<td>5.44%</td>
<td>4.22% </td>
</tr>
<tr>
<td>D.S2</td>
<td>Klink</td>
<td>24.29%</td>
<td>1.25%</td>
<td>2.38%</td>
<td>25.68%</td>
<td>3.57%</td>
<td>6.27% </td>
</tr>
<tr>
<td>D.S3</td>
<td>qwen2.5:7b</td>
<td>20.65%</td>
<td>2.72%</td>
<td>4.80%</td>
<td>17.72%</td>
<td>6.29%</td>
<td>9.28% </td>
</tr>
<tr>
<td>D.S3</td>
<td>qwen2.5:32b</td>
<td>49.39%</td>
<td>3.04%</td>
<td>5.73%</td>
<td>39.34%</td>
<td>6.53%</td>
<td>11.19% </td>
</tr>
<tr>
<td>D.S3</td>
<td>qwen2.5:72b</td>
<td>51.42%</td>
<td>3.10%</td>
<td>5.85%</td>
<td>42.49%</td>
<td>6.91%</td>
<td>11.88% </td>
</tr>
<tr>
<td>D.S3</td>
<td>llama3.2:3b</td>
<td>45.75%</td>
<td>0.48%</td>
<td>0.96%</td>
<td>48.50%</td>
<td>1.38%</td>
<td>2.69% </td>
</tr>
<tr>
<td>D.S3</td>
<td>llama3.1:70b</td>
<td>31.98%</td>
<td>2.08%</td>
<td>3.90%</td>
<td>24.77%</td>
<td>4.34%</td>
<td>7.39% </td>
</tr>
<tr>
<td>D.S3</td>
<td>deepseek-r1:7b</td>
<td>25.51%</td>
<td>0.49%</td>
<td>0.97%</td>
<td>27.18%</td>
<td>1.41%</td>
<td>2.69% </td>
</tr>
<tr>
<td>D.S3</td>
<td>glm4:9b</td>
<td>43.72%</td>
<td>0.82%</td>
<td>1.61%</td>
<td>40.69%</td>
<td>2.06%</td>
<td>3.92% </td>
</tr>
<tr>
<td>D.S3</td>
<td>llama3.3:70b</td>
<td>55.06%</td>
<td>1.21%</td>
<td>2.37%</td>
<td>50.00%</td>
<td>2.97%</td>
<td>5.61% </td>
</tr>
<tr>
<td>D.S3</td>
<td>deepseek-r1:32b</td>
<td>64.78%</td>
<td>2.05%</td>
<td>3.98%</td>
<td>52.55%</td>
<td>4.49%</td>
<td>8.28% </td>
</tr>
<tr>
<td>D.S3</td>
<td>deepseek-r1:70b</td>
<td>62.35%</td>
<td>1.80%</td>
<td>3.51%</td>
<td>53.75%</td>
<td>4.20%</td>
<td>7.78% </td>
</tr>
</table>
</div>