export async function playAudioOnce(
  src: string | Blob | ArrayBuffer,
  opts: { volume?: number } = {}
): Promise<void> {
  const { volume = 1 } = opts;

  let objectUrl: string | null = null;

  try {
    if (src instanceof ArrayBuffer) {
      src = new Blob([src], { type: "audio/wav" });
    }
    const url =
      typeof src === "string" ? src : (objectUrl = URL.createObjectURL(src));

    await new Promise<void>((resolve, reject) => {
      const audio = new Audio();
      audio.src = url;
      audio.volume = volume;
      audio.loop = false;
      audio.preload = "auto";
      audio.crossOrigin = "anonymous";
      audio.currentTime = 0;

      const cleanup = () => {
        audio.onended = null;
        audio.onerror = null;
        if (objectUrl) {
          URL.revokeObjectURL(objectUrl);
          objectUrl = null;
        }
      };

      audio.onended = () => {
        cleanup();
        resolve();
      };
      audio.onerror = () => {
        const err = new Error("Failed to play audio.");
        cleanup();
        reject(err);
      };

      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch((e) => {
          cleanup();
          reject(e);
        });
      }
    });
  } finally {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
}
