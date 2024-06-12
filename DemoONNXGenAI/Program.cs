using Microsoft.ML.OnnxRuntimeGenAI;


var modelDirectory = @"E:/models/Phi-3-vision-128k-instruct-onnx-cpu/cpu-int4-rtn-block-32-acc-level-4";

using var model = new Model(modelDirectory);
using MultiModalProcessor processor = new MultiModalProcessor(model);
using var tokenizerStream = processor.CreateStream();

while (true)
{
    Console.Write("Image Path (leave empty if no image): ");
    var imagePath = Console.ReadLine();
    var hasImage = !string.IsNullOrWhiteSpace(imagePath);

    Console.WriteLine(hasImage ? "Loading image..." : "No image");
    Images? images = hasImage ? Images.Load(imagePath) : null;

    Console.Write("Prompt: ");
    var text = Console.ReadLine();
    if (text == null) { continue; }

    var prompt = "<|user|>\n";
    prompt += hasImage ? "<|image_1|>\n" : "";
    prompt += text + "<|end|>\n<|assistant|>\n";

    Console.WriteLine($"Processing...");
    using var inputs = processor.ProcessImages(prompt, images);

    Console.WriteLine($"Generating response...");
    using var generatorParams = new GeneratorParams(model);
    generatorParams.SetInputs(inputs);
    generatorParams.SetSearchOption("max_length", 3072);
    using var generator = new Generator(model, generatorParams);

    Console.WriteLine("================  Output  ================");
    while (!generator.IsDone())
    {
        generator.ComputeLogits();
        generator.GenerateNextToken();
        var newTokens = generator.GetSequence(0);
        var output = tokenizerStream.Decode(newTokens[^1]);
        Console.Write(output);
    }
    Console.WriteLine();
    Console.WriteLine("==========================================");

    images?.Dispose();
}