import re
import torch


class Text2SQLMasker:
    def __init__(self):
        # Your valid SQL keyword set
        self.keywords = {
            'FULL OUTER JOIN', 'OR', 'EXCEPT', 'OF', 'DISTINCT', 'SELECT', 'FALSE', 'ASC',
            'DESC', 'USING', 'YEAR', 'INTO', 'BETWEEN', 'OFFSET', 'INTERSECT', 'INSERT',
            'JOIN', 'LEFT JOIN', 'UNION', 'LIMIT', 'AND', 'INNER JOIN', 'RIGHT JOIN', 'TRUE',
            'LEFT OUTER JOIN', 'AS', 'GROUP BY', 'IN', 'WINDOW', 'CROSS JOIN', 'ON',
            'ORDER BY', 'RETURNING', 'HAVING', 'FROM', 'UNION ALL', '(', ")", '=', '.', ';', ',',
            '<', '>', '!=', '<=', '>=', '+', '-', '*', '/', '%', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
            'LIKE', 'IS', 'NULL', 'NOT', 'CREATE', 'TABLE', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'WHERE'
        }
        # Filled from schema/context (e.g. column names, table names)
        self.context_words = set()

        # Internal cache of tokenized phrases
        self._phrase_token_ids = None

    def set_context_words(self, sql_context: str):
        """
        Optionally populate context_words from a SQL context string.
        This is just an example that extracts column names from INSERT ... (col1, col2, ...).
        """
        context_tokens = set()

        match = re.findall(r'(?<=INSERT INTO).*?\((.*?)\)', sql_context)
        for grp in match:
            columns = set(grp.split(', '))
            context_tokens |= columns

        self.context_words = context_tokens
        self._phrase_token_ids = None  # reset cache when context changes

    def _build_phrase_token_ids(self, tokenizer):
        """
        Tokenize all valid phrases (keywords + context words) once and cache them.
        """
        phrases = list(self.keywords | self.context_words)
        phrase_token_ids = []

        for phrase in phrases:
            # Leading space matches normal BPE/SPM behavior
            ids = tokenizer.encode(" " + phrase, add_special_tokens=False)
            if len(ids) > 0:
                phrase_token_ids.append(tuple(ids))

        self._phrase_token_ids = phrase_token_ids

    def get_masking_confidence_scores(self, x0: torch.Tensor, tokenizer) -> torch.Tensor:
        """
        Compute confidence scores for the masking schedule.

        Behavior:
        - Tokens that are part of any valid SQL phrase get a random score in [0, 1).
        - All other tokens get a very low score (-1e9), so they are selected
          for unmasking only at the very end (after being regenerated many times).

        Args:
            x0: (batch, seq_len) tensor of current argmax predictions (token IDs).
            tokenizer: tokenizer used to encode the SQL phrases.

        Returns:
            x0_p: (batch, seq_len) float tensor of "confidence" scores.
        """
        if self._phrase_token_ids is None:
            self._build_phrase_token_ids(tokenizer)

        device = x0.device
        batch_size, seq_len = x0.shape

        VERY_LOW = -1e9
        # Start by assuming everything is "non-SQL" (very low confidence)
        x0_p = torch.full(
            (batch_size, seq_len),
            VERY_LOW,
            device=device,
            dtype=torch.float32,
        )
        print("getting confidence for Text2Sql")
        phrase_token_ids = self._phrase_token_ids

        for b in range(batch_size):
            seq = x0[b].tolist()
            sql_positions = set()

            # Find all positions that belong to any SQL phrase
            for phrase_ids in phrase_token_ids:
                n = len(phrase_ids)
                if n == 0 or n > seq_len:
                    continue

                # Simple sliding-window match
                for i in range(seq_len - n + 1):
                    if seq[i:i + n] == list(phrase_ids):
                        for k in range(n):
                            sql_positions.add(i + k)
                            print(f"Found SQL token at position {i + k} in batch {b}")

            if sql_positions:
                sql_positions = sorted(sql_positions)
                # Assign random confidence in [0, 1) to *only* SQL positions
                rand_vals = torch.rand(len(sql_positions), device=device)
                for idx, pos in enumerate(sql_positions):
                    x0_p[b, pos] = rand_vals[idx]

        return x0_p
