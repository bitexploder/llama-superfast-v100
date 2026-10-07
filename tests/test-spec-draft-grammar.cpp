// [TAG_SPEC_REJECTION] a drafted token accepted into a copy of the target's sampler must not throw when
// it does not fit the copy's triggered grammar. The draft proposes its argmax where the copy cannot follow the target,
// and after a tool call's closing tag the draft's <|im_start|> does not fit: common_sampler_accept threw "Unexpected
// empty grammar stack", which failed every tool-calling request under LLAMA_SPEC_REJECTION. Checks, on a vocab-only
// model (argv[1]) with a lazy tool-call grammar:
//   1. the control: common_sampler_accept throws on that token (the failure the server hit);
//   2. common_sampler_accept_draft takes it without throwing;
//   3. a token that fits still advances the grammar under common_sampler_accept_draft;
//   4. before the trigger (the lazy grammar awaiting it) any token is taken, and the trigger token triggers it.

#include "common.h"
#include "sampling.h"

#include "llama.h"

#include <cstdio>
#include <string>

static int n_fail = 0;

static void check(bool ok, const char * what) {
    printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    n_fail += ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <vocab gguf>\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto tok = [&](const std::string & s) { return common_tokenize(vocab, s, false, true); };

    const auto open = tok("<tool_call>");
    const auto im_end = tok("<|im_end|>");
    const auto im_start = tok("<|im_start|>");
    if (im_end.size() != 1 || im_start.size() != 1) {
        fprintf(stderr, "the vocab lacks the special tokens\n");
        return 1;
    }

    common_params_sampling sp;
    sp.grammar      = common_grammar(COMMON_GRAMMAR_TYPE_USER, "root ::= \"<tool_call>\" [a-z]+ \"</tool_call>\"");
    sp.grammar_lazy = true;
    // a word trigger: the test vocabs spell <tool_call> in several tokens (the model's chat format triggers on its
    // single token; after the trigger the grammar is the same)
    sp.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>", LLAMA_TOKEN_NULL });

    common_sampler * base = common_sampler_init(model, sp);
    check(base != nullptr, "sampler with a lazy tool-call grammar");
    if (!base) {
        return 1;
    }
    const auto accept_all = [](common_sampler * s, const llama_tokens & t, bool draft) {
        for (auto x : t) {
            if (draft) {
                common_sampler_accept_draft(s, x);
            } else {
                common_sampler_accept(s, x, true);
            }
        }
    };
    const auto throws = [&](common_sampler * s, const llama_tokens & t) {
        try {
            accept_all(s, t, false);
        } catch (const std::exception &) {
            return true;
        }
        return false;
    };

    // 4. before the trigger: any token, through either call
    bool ok = true;
    try {
        accept_all(base, tok("Let me check the weather."), true);
        accept_all(base, im_start, true);
    } catch (const std::exception &) {
        ok = false;
    }
    check(ok, "before the trigger, any drafted token is taken");
    accept_all(base, open, true);
    {
        common_sampler * c = common_sampler_clone(base);
        check(throws(c, im_start), "after the trigger the grammar applies (control: a token that does not fit throws)");
        common_sampler_free(c);
    }

    // the call, then the end of turn
    accept_all(base, tok("paris"), false);
    accept_all(base, tok("</tool_call>"), false);
    accept_all(base, im_end, false);

    // 1. control and 2. the fix
    {
        common_sampler * c = common_sampler_clone(base);
        check(throws(c, im_start), "control: common_sampler_accept throws on <|im_start|> after the call");
        common_sampler_free(c);
    }
    {
        common_sampler * c = common_sampler_clone(base);
        ok = true;
        try {
            accept_all(c, im_start, true);
            accept_all(c, tok("assistant"), true);
        } catch (const std::exception &) {
            ok = false;
        }
        check(ok, "common_sampler_accept_draft takes <|im_start|> and more after the call");
        common_sampler_free(c);
    }

    // 3. a fitting token advances the grammar: after the trigger and "abc", "</tool_call>" must fit (it would not if
    // "abc" had been kept from the grammar, which then still wants [a-z])
    {
        common_sampler * s = common_sampler_init(model, sp);
        accept_all(s, open, true);
        accept_all(s, tok("abc"), true);
        common_sampler * c = common_sampler_clone(s);
        check(!throws(c, tok("</tool_call>")), "a fitting drafted token advances the grammar");
        common_sampler_free(c);
        common_sampler * d = common_sampler_clone(s);
        check(throws(d, im_start), "control: the grammar still applies after the fitting tokens");
        common_sampler_free(d);
        common_sampler_free(s);
    }

    common_sampler_free(base);
    llama_model_free(model);
    printf("%s\n", n_fail == 0 ? "all passed" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
