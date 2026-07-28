"""Grammar-constrained decoding for tool-call generation.

Constrains the decoder to only produce valid tool names and argument keys
by tracking position in the output JSON and masking invalid tokens via a
character-level trie built from the tool definitions.

Needle output format (compact JSON, no spaces):
  [{"name":"tool_name","arguments":{"key1":value1,"key2":value2}}]

Constrained regions:
  - Tool names after "name":" are constrained to known tool names
  - Argument keys after "arguments":{ are constrained to known param names
  - Argument VALUES of parameters with declared enums (an "enum" list in the
    schema, or an out-of-band value_enums dict) are constrained to the legal
    values, INCLUDING the opening quote (so non-string garbage is impossible).
    Values of parameters without enums remain unconstrained.
"""

import json
import logging
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class TrieNode:
    __slots__ = ("children", "is_terminal")

    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_terminal: bool = False


class Trie:
    """Character-level prefix tree for matching valid names/keys."""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_terminal = True

    def get_node(self, prefix: str) -> TrieNode | None:
        """Walk the trie for *prefix* and return the node, or None."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    @property
    def words(self) -> list[str]:
        """Return all words stored in the trie."""
        result = []
        def _dfs(node, path):
            if node.is_terminal:
                result.append("".join(path))
            for ch, child in sorted(node.children.items()):
                _dfs(child, path + [ch])
        _dfs(self.root, [])
        return result


class ToolConstraints:
    """Holds one name trie and one param trie per function, built from tool JSON.

    Extracts property names from JSON Schema ``parameters.properties``,
    not from the schema-level keys (``type``, ``properties``, ``required``).
    """

    def __init__(self, tools_json: str, value_enums: dict | None = None):
        """value_enums: optional out-of-band {param_name: [legal values]} applied
        to every function that has that parameter (e.g. dynamic device-side sets:
        contacts, playlists, rooms). Schema-declared "enum" lists are also read."""
        self.name_trie = Trie()
        self.param_tries: dict[str, Trie] = {}
        self.value_tries: dict[tuple[str, str], Trie] = {}

        try:
            tools = json.loads(tools_json)
        except (json.JSONDecodeError, TypeError):
            tools = []

        if not isinstance(tools, list):
            tools = []

        shared_tries: dict[str, Trie] = {}       # out-of-band enums, built once per key
        if value_enums:
            for key, vals in value_enums.items():
                vt = Trie()
                for v in vals:
                    if isinstance(v, str) and v:
                        vt.insert(v)
                if vt.root.children:
                    shared_tries[key] = vt
        # kept separately so value constraints survive even when the function
        # name was never parsed (malformed/unknown name -> current_function "")
        self.shared_value_tries = shared_tries

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name", "")
            if not name:
                continue
            self.name_trie.insert(name)

            # device-known enums apply GLOBALLY by param name - even if the model
            # hallucinates the param onto a tool that doesn't declare it, the
            # value stays legal (the executor then rejects the unknown arg)
            for key, vt in shared_tries.items():
                self.value_tries[(name, key)] = vt

            params = tool.get("parameters", {})
            if isinstance(params, dict):
                param_trie = Trie()
                for key, val in params.items():
                    if isinstance(val, dict):
                        param_trie.insert(key)
                        enum_vals = val.get("enum")
                        if enum_vals:
                            vt = Trie()
                            for v in enum_vals:
                                if isinstance(v, str) and v:
                                    vt.insert(v)
                            if vt.root.children:
                                self.value_tries[(name, key)] = vt
                self.param_tries[name] = param_trie

    def get_param_trie(self, function_name: str) -> Trie | None:
        return self.param_tries.get(function_name)

    def get_value_trie(self, function_name: str, param: str) -> Trie | None:
        vt = self.value_tries.get((function_name, param))
        return vt if vt is not None else self.shared_value_tries.get(param)


class JsonState(Enum):
    FREE = auto()
    IN_NAME = auto()
    IN_ARG_KEY = auto()
    AWAIT_VALUE = auto()   # after "<enum-key>": -> next char MUST open a legal string value
    IN_ARG_VALUE = auto()  # inside the quotes of an enum-constrained value


class JsonStateMachine:
    """Tracks position in needle's compact JSON output to constrain decoding.

    Needle format: ``[{"name":"TOOL","arguments":{"key":val,...}},...]``

    Constrained spans:
      - After ``"name":"`` → IN_NAME (constrains to valid tool names)
      - After ``"arguments":{"`` or ``,"`` at arguments depth → IN_ARG_KEY
        (constrains to valid parameter names for the current tool)
      - Closing ``"`` in constrained state → FREE

    Depth tracking ensures nested objects/arrays in argument VALUES
    do not trigger false IN_ARG_KEY transitions.
    """

    def __init__(self, tool_constraints: "ToolConstraints | None" = None):
        self.state = JsonState.FREE
        self.buffer = ""
        self.constrained_buf = ""
        self.current_function = ""
        self.current_key = ""
        self.in_arguments = False
        self.arguments_depth = 0
        self.nesting_depth = 0
        self.in_string = False
        self.prev_char_escape = False
        self.val_escape = False
        self.expect_colon = False     # an enum'd key just closed; next char must be ':'
        self.tc = tool_constraints    # needed to know which keys have value enums

    def _value_trie(self) -> "Trie | None":
        if self.tc is None:
            return None
        return self.tc.get_value_trie(self.current_function, self.current_key)

    def feed(self, text: str):
        """Feed generated text character-by-character to drive transitions."""
        for ch in text:
            self._feed_char(ch)

    def _feed_char(self, ch: str):
        if self.state in (JsonState.IN_NAME, JsonState.IN_ARG_KEY):
            if ch == '"':
                if self.state == JsonState.IN_NAME:
                    self.current_function = self.constrained_buf
                else:
                    self.current_key = self.constrained_buf
                    self.expect_colon = self._value_trie() is not None
                self.constrained_buf = ""
                self.state = JsonState.FREE
            else:
                self.constrained_buf += ch
            self.buffer += ch
            return

        if self.state == JsonState.IN_ARG_VALUE:
            self.buffer += ch
            if self.val_escape:
                self.val_escape = False
                self.constrained_buf += ch
                return
            if ch == '\\':
                self.val_escape = True
                self.constrained_buf += ch
                return
            if ch == '"':
                self.constrained_buf = ""
                self.state = JsonState.FREE
                return
            self.constrained_buf += ch
            return

        if self.state == JsonState.AWAIT_VALUE:
            if ch == '"':
                self.buffer += ch
                self.constrained_buf = ""
                self.val_escape = False
                self.state = JsonState.IN_ARG_VALUE
                return
            self.state = JsonState.FREE      # fallback emitted a non-string value:
            # fall through to normal FREE processing of this char

        if self.expect_colon:                # an enum'd key just closed; ':' is next
            self.expect_colon = False
            if ch == ':':
                self.buffer += ch
                self.state = JsonState.AWAIT_VALUE
                return
            # anything else: malformed JSON, give up on constraining this value

        self.buffer += ch

        if self.in_string:
            if self.prev_char_escape:
                self.prev_char_escape = False
                return
            if ch == '\\':
                self.prev_char_escape = True
                return
            if ch == '"':
                self.in_string = False
            return

        if ch in '{[':
            self.nesting_depth += 1
        elif ch in '}]':
            self.nesting_depth = max(0, self.nesting_depth - 1)
            if ch == '}' and self.in_arguments and self.nesting_depth < self.arguments_depth:
                self.in_arguments = False
            return

        if self.buffer.endswith('"name":"') and not self.in_arguments:
            self.state = JsonState.IN_NAME
            self.constrained_buf = ""
            return

        if self.buffer.endswith('"arguments":{'):
            self.in_arguments = True
            self.arguments_depth = self.nesting_depth
            return

        if (self.in_arguments
                and self.nesting_depth == self.arguments_depth
                and self._at_arg_key_start()):
            self.state = JsonState.IN_ARG_KEY
            self.constrained_buf = ""
            return

        if ch == '"' and self._is_value_quote():
            self.in_string = True

    def _at_arg_key_start(self) -> bool:
        """True if buffer ends with ``{"`` or ``,"`` — an arg key is opening."""
        if len(self.buffer) < 2:
            return False
        return self.buffer[-2:] in ('{"', ',"')

    def _is_value_quote(self) -> bool:
        """True if the current ``"`` opens a JSON string value (preceded by ``:``)."""
        for j in range(len(self.buffer) - 2, -1, -1):
            c = self.buffer[j]
            if c in ' \t\n\r':
                continue
            return c == ':'
        return False


def build_token_strings(tokenizer) -> list[str]:
    """Map each vocab ID to the exact characters it contributes to output.

    SentencePiece uses ``▁`` (U+2581) as a word-boundary marker that becomes
    a space in decoded text.  We convert ``▁`` → ``' '`` so that both the
    state machine and the trie matcher see consistent text.  This correctly
    blocks ``▁``-prefixed tokens inside constrained spans (tool names and
    argument keys never contain spaces in needle's compact JSON format).
    """
    vocab_size = tokenizer.vocab_size
    sp = tokenizer.sp
    strings = []
    for i in range(vocab_size):
        if sp.IsControl(i):
            strings.append("")
        elif sp.IsUnknown(i):
            strings.append("")
        elif sp.IsByte(i):
            piece = sp.IdToPiece(i)
            try:
                byte_val = int(piece[1:-1], 16) if piece.startswith("<0x") else ord(piece)
                strings.append(chr(byte_val))
            except (ValueError, IndexError):
                strings.append("")
        else:
            piece = sp.IdToPiece(i)
            strings.append(piece.replace("\u2581", " "))
    return strings


class TokenIndex:
    """Maps first character -> list of token IDs for fast candidate filtering."""

    def __init__(self, token_strings: list[str]):
        self._index: dict[str, list[int]] = {}
        for tid, s in enumerate(token_strings):
            if not s:
                continue
            first = s[0]
            if first not in self._index:
                self._index[first] = []
            self._index[first].append(tid)

    def candidates_for(self, first_char: str) -> list[int]:
        return self._index.get(first_char, [])

    @property
    def all_nonempty(self) -> list[int]:
        """All token IDs with non-empty strings."""
        result = []
        for ids in self._index.values():
            result.extend(ids)
        return result


def _check_token_valid(token_text: str, trie_node: TrieNode) -> bool:
    """Check if *token_text* is a valid continuation from *trie_node*.

    The token text may contain a closing ``"`` which signals end of the
    constrained span — at that point the trie node must be terminal.
    Characters after the closing ``"`` are not checked (they are structural
    JSON that the state machine handles separately).
    """
    node = trie_node
    for i, ch in enumerate(token_text):
        if ch == '"':
            return node.is_terminal
        if ch not in node.children:
            return False
        node = node.children[ch]
    return True


def _walk_value_path(text: str, trie_root: TrieNode) -> bool:
    """Validate chars that follow an enum'd key's closing quote:
    ``:`` then ``"`` then a value-trie walk; a further ``"`` requires a
    terminal node (value complete; the structural tail is unchecked).
    Any prefix of this path is valid (the token may stop anywhere)."""
    if not text:
        return True
    if text[0] != ':':
        return False
    rest = text[1:]
    if not rest:
        return True
    if rest[0] != '"':
        return False
    node = trie_root
    for ch in rest[1:]:
        if ch == '"':
            return node.is_terminal
        if ch not in node.children:
            return False
        node = node.children[ch]
    return True


def _check_token_valid_key(token_text: str, key_node: TrieNode, key_buf: str,
                           get_value_trie) -> bool:
    """IN_ARG_KEY token check that also validates any VALUE characters the
    token smuggles past the key's closing quote (``ion":"gar`` etc.)."""
    node = key_node
    consumed = []
    for i, ch in enumerate(token_text):
        if ch == '"':
            if not node.is_terminal:
                return False
            vtrie = get_value_trie(key_buf + "".join(consumed))
            if vtrie is None:
                return True                     # no enum: tail unchecked (as before)
            return _walk_value_path(token_text[i + 1:], vtrie.root)
        if ch not in node.children:
            return False
        node = node.children[ch]
        consumed.append(ch)
    return True


def _check_token_valid_value_open(token_text: str, trie_root: TrieNode) -> bool:
    """Valid opener for an enum'd value: a leading ``"`` then a walk of the
    value trie; a SECOND ``"`` inside the token closes the value and requires
    a terminal node (whole values in one token, e.g. ``"on"``)."""
    if not token_text or token_text[0] != '"':
        return False
    node = trie_root
    for ch in token_text[1:]:
        if ch == '"':
            return node.is_terminal
        if ch not in node.children:
            return False
        node = node.children[ch]
    return True


def apply_value_open_constraints(
    logits: np.ndarray,
    trie_root: TrieNode,
    token_strings: list[str],
    token_index: TokenIndex,
) -> np.ndarray:
    """Mask logits in AWAIT_VALUE: the next token MUST open a legal enum value."""
    vocab_size = logits.shape[0]
    mask = np.full(vocab_size, False)
    for tid in token_index.candidates_for('"'):
        if _check_token_valid_value_open(token_strings[tid], trie_root):
            mask[tid] = True
    if not mask.any():
        logger.warning("Constrained decoding: no valid value-opening tokens, falling back")
        return logits
    masked = logits.copy()
    masked[~mask] = -np.inf
    return masked


def apply_constraints(
    logits: np.ndarray,
    state: JsonState,
    trie_node: TrieNode,
    token_strings: list[str],
    token_index: TokenIndex,
) -> np.ndarray:
    """Mask logits so only valid tokens survive.

    Args:
        logits: shape (vocab_size,) float array
        state: current constrained state (IN_NAME or IN_ARG_KEY)
        trie_node: current position in the relevant trie
        token_strings: mapping from token ID to text
        token_index: first-char index for fast filtering

    Returns:
        Modified logits with invalid tokens set to -inf.
    """
    vocab_size = logits.shape[0]
    mask = np.full(vocab_size, False)

    valid_first_chars = set(trie_node.children.keys())
    if trie_node.is_terminal:
        valid_first_chars.add('"')

    for first_char in valid_first_chars:
        for tid in token_index.candidates_for(first_char):
            if not mask[tid]:
                text = token_strings[tid]
                if _check_token_valid(text, trie_node):
                    mask[tid] = True

    if not mask.any():
        logger.warning("Constrained decoding: no valid tokens found, falling back to unconstrained")
        return logits

    masked_logits = logits.copy()
    masked_logits[~mask] = -np.inf
    return masked_logits

class ConstrainedDecoder:
    """Batch-aware constrained decoder.

    Holds per-example ``JsonStateMachine`` instances and shared token
    metadata. Call ``constrain_logits`` between logit computation and
    argmax, then ``update`` after selecting the token.

    Both ``constrain_logits`` and ``update`` use the same ``token_strings``
    table (with ``▁`` → space) so the state machine and trie matcher see
    identical text.  This correctly blocks ``▁``-prefixed tokens inside
    constrained spans, since spaces are never valid in tool names or
    argument keys.
    """

    def __init__(self, tool_constraints_list: list[ToolConstraints],
                 token_strings: list[str], token_index: TokenIndex,
                 tokenizer=None):
        self.batch_size = len(tool_constraints_list)
        self.tool_constraints = tool_constraints_list
        self.machines = [JsonStateMachine(tc) for tc in tool_constraints_list]
        self.token_strings = token_strings
        self.token_index = token_index

    def is_active(self, batch_idx: int) -> bool:
        """Return True if this batch element is currently in a constrained state."""
        m = self.machines[batch_idx]
        return m.state != JsonState.FREE or (m.expect_colon and m._value_trie() is not None)

    def constrain_logits(self, logits: np.ndarray, batch_idx: int) -> np.ndarray:
        """Apply grammar constraints to logits for a single batch element."""
        machine = self.machines[batch_idx]
        tc = self.tool_constraints[batch_idx]

        if machine.state == JsonState.FREE:
            if machine.expect_colon:            # colon-crossing tokens (':"ga...') get
                vtrie = machine._value_trie()   # validated through the whole value path
                if vtrie is None:
                    return logits
                vocab_size = logits.shape[0]
                mask = np.full(vocab_size, False)
                for tid in self.token_index.candidates_for(':'):
                    if _walk_value_path(self.token_strings[tid], vtrie.root):
                        mask[tid] = True
                if not mask.any():
                    logger.warning("Constrained decoding: no valid colon tokens, falling back")
                    return logits
                masked = logits.copy()
                masked[~mask] = -np.inf
                return masked
            return logits

        if machine.state == JsonState.AWAIT_VALUE:
            vtrie = machine._value_trie()
            if vtrie is None:
                return logits
            return apply_value_open_constraints(logits, vtrie.root,
                                                self.token_strings, self.token_index)

        if machine.state == JsonState.IN_NAME:
            trie = tc.name_trie
        elif machine.state == JsonState.IN_ARG_KEY:
            trie = tc.get_param_trie(machine.current_function)
            if trie is None:
                return logits
            node = trie.get_node(machine.constrained_buf)
            if node is None:
                logger.warning("Constrained decoding: off-trie at %r, falling back", machine.constrained_buf)
                return logits
            # key tokens may smuggle value chars past the closing quote -> full-path check
            get_vt = lambda key: tc.get_value_trie(machine.current_function, key)
            vocab_size = logits.shape[0]
            mask = np.full(vocab_size, False)
            first_chars = set(node.children.keys())
            if node.is_terminal:
                first_chars.add('"')
            for fc in first_chars:
                for tid in self.token_index.candidates_for(fc):
                    if not mask[tid] and _check_token_valid_key(
                            self.token_strings[tid], node, machine.constrained_buf, get_vt):
                        mask[tid] = True
            if not mask.any():
                logger.warning("Constrained decoding: no valid key tokens, falling back")
                return logits
            masked = logits.copy()
            masked[~mask] = -np.inf
            return masked
        elif machine.state == JsonState.IN_ARG_VALUE:
            trie = machine._value_trie()
            if trie is None:
                return logits
        else:
            return logits

        node = trie.get_node(machine.constrained_buf)
        if node is None:
            logger.warning("Constrained decoding: off-trie at %r, falling back", machine.constrained_buf)
            return logits

        return apply_constraints(logits, machine.state, node, self.token_strings, self.token_index)

    def update(self, batch_idx: int, token_id: int):
        """Advance the state machine for *batch_idx* with the selected token."""
        text = self.token_strings[token_id]
        self.machines[batch_idx].feed(text)


_token_cache: dict[int, tuple[list[str], TokenIndex]] = {}


def _get_token_data(tokenizer) -> tuple[list[str], TokenIndex]:
    """Return (token_strings, token_index), cached per tokenizer instance."""
    key = id(tokenizer)
    if key not in _token_cache:
        ts = build_token_strings(tokenizer)
        _token_cache[key] = (ts, TokenIndex(ts))
    return _token_cache[key]


def build_constrained_decoder(
    tools_json_list: list[str],
    tokenizer,
    value_enums: dict | None = None,
) -> ConstrainedDecoder:
    """Convenience factory: build a ConstrainedDecoder for a batch of examples.

    Args:
        tools_json_list: list of tool JSON strings (one per batch element)
        tokenizer: NeedleTokenizer with .sp SentencePiece model
        value_enums: optional {param_name: [legal values]} - device-side closed
            sets (contacts, rooms, playlists); schema "enum" fields also work
    """
    token_strings, token_index = _get_token_data(tokenizer)
    tc_list = [ToolConstraints(tj, value_enums=value_enums) for tj in tools_json_list]
    return ConstrainedDecoder(tc_list, token_strings, token_index)
