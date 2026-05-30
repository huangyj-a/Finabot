# import akshare as ak

# stock_sse_summary_df = ak.stock_szse_summary(date="20260529")
# print(stock_sse_summary_df)

# import akshare as ak

# top_20_index = [
#     "上证指数", "深证成指", "创业板指", "科创50", "沪深300",
#     "上证50", "中证500", "中证1000", "北证50", "深证100",
#     "上证180", "中证2000", "中证全指", "创业板50", "上证科创板50",
#     "中证A50", "中证A500", "中证800", "中小100", "富时A50指数",
#     "国债指数", "沪企债指数", "深企债指数", "上证国债指数", "中证国债指数",
#     "中证红利", "深证红利", "沪深300价值", "中证医疗", "中证消费"
# ]
# stock_zh_index_spot_sina_df = ak.stock_zh_index_spot_sina()
# # 筛选并打印
# result = stock_zh_index_spot_sina_df[stock_zh_index_spot_sina_df["名称"].isin(top_20_index)]
# print(result)

# import akshare as ak

# # 获取港股所有指数实时数据
# df = ak.stock_hk_index_spot_sina()

# # 港股最经典十大指数（按市场关注度排序）
# classic_hk_index = [
#     "恒生指数",         # 港股大盘基准
#     "恒生科技指数",     # 港股科技龙头
#     "恒生中国企业指数", # H股国企指数
#     "恒生香港中资企业指数",
#     "恒生互联网科技业指数",
#     "恒生医疗保健指数",
#     "恒生金融业指数",
#     "恒生地产建筑业指数",
#     "恒生消费品制造及服务业指数",
#     "恒生高股息率指数"
# ]

# # 筛选并打印
# result = df[df["名称"].isin(classic_hk_index)]
# print(result)


# import akshare as ak

# stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000001")

# # 最近100天数据
# last_100_df = stock_zh_index_daily_df.tail(100)
# print(last_100_df)