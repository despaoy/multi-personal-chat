# 20 条过拟合链路测试

固定测试数据和配置已经生成。服务器具备基础模型和 GPU 时执行：

```bash
python scripts/run_kisaki_v4_overfit_test.py
```

命令会依次完成短跑训练、适配器生成和审核包渲染。人工只审核随后生成的 `review.md` 中的 20 条模型回答。
