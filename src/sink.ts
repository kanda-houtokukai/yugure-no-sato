// dev サーバ側の受け口（vite.config.ts の sinkPlugin）へ JSON を送る。
// Code が「ブラウザの中で起きたこと」をファイルとして受け取るための唯一の経路。
// 本番ビルドでは受け口が存在しないので、失敗しても黙って無視する。

export async function postToSink(name: string, data: unknown): Promise<boolean> {
  try {
    const res = await fetch(`/__sink/${encodeURIComponent(name)}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(data, null, 2),
    });
    return res.ok;
  } catch {
    return false;
  }
}
