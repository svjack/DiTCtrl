This is the prompts generation instruction for Presto, which is modified from the Presto's [arxiv paper](https://arxiv.org/pdf/2412.01316). You can copy and paste it to Large Language Model (e.g., GPT-4) to generate your own prompts. Usually, you should define the video content description <VIDEO_DESCRIPTION> first, and then use the prompts for our task.

```
System Prompt:
You are a helpful video director.

User Prompt:
Based on the video content description, you need to write three coherent scene descriptions to create a silent video. These three descriptions represent consecutive moments within the same scene, each lasting approximately 49 frames (about 3 seconds). The three descriptions are connected, representing natural temporal progression within a single scenario.

These three scene descriptions should include detailed scenario transitions (such as camera movement, background changes, and object movement). The camera movement should be smooth. Avoid drastic angle changes and transitions, such as shifting from a frontal view directly to a side view. You can add details and objects, but the three scenes must form a continuous story, which means repeated object descriptions and details may be omitted. Three scene descriptions should NOT differ too much. Ensure similarity to enable smooth transitions between scenes. 

If the description is brief, you can add details, but stay conservative, and only create simple, easily generated scenes. It's also acceptable for multiple scenes to share a higher degree of similarity. Since these are temporally close moments within the same scene (approximately 9 seconds total), they should maintain high visual consistency with natural, subtle changes.

You need to accurately, objectively, and succinctly describe everything. The scene descriptions need to be concise. Do NOT add details unrelated to the video content description. Do NOT speculate. Do NOT add scene titles, directly return three scene descriptions.

The video content description: <VIDEO_DESCRIPTION>.
```