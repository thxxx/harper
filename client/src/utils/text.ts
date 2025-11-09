export function extractBracketTexts(input: string): string[] {
  // [] 안의 내용을 캡처
  const regex = /\[(.*?)\]/g;
  const matches = input.match(regex);

  if (!matches) return [];

  // 양쪽 대괄호 제거 후 리턴
  return matches.map((m) => m.slice(1, -1));
}
// 결과: ["is", "sample", "string"]
