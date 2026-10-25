import sys
sys.path.insert(0, "/home/erokhins/erokhins_projects/qwen3.6-35b-a3b/tools")
import audit_published_quant as apq, gguf_reader as gr
gr.grant_selftest_token(apq.SELFTEST_TOKEN)
gg = gr.Gguf(sys.argv[1])
n = [t for t in gg.tensors if "shexp" in t.name]
print("tensors_with_shexp=%d of %d" % (len(n), len(gg.tensors)))
for t in n[:5]:
    print("  name=%s type=%s dims=%s row_bytes=%d" % (t.name, t.type_name, t.dims, t.row))
