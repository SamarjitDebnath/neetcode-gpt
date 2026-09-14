from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        
        def _get_stats(tokens):
            pairs = {}
            for pair in zip(tokens, tokens[1:]):
                pairs[pair] = pairs.get(pair, 0) + 1
            return pairs

        def _get_newTokens(tokens, best):
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < (len(tokens) - 1)) and (tokens[i] == best[0]) and (tokens[i+1] == best[1]):
                    new_tokens.append(best[0] + best[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            return new_tokens

        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break
            
            # get best pairs
            pairs = _get_stats(tokens)
            if not pairs:
                break

            best_count = max(pairs.values())
            candidates = sorted(p for p, c in pairs.items() if c == best_count)
            best = candidates[0]
            merges.append(best)

            tokens = _get_newTokens(tokens, best)

        return merges








                
