import 'dart:developer';

import 'package:flutter/services.dart';
import 'package:mlfacerecognition/imagecovertion.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'detector_painters.dart';
import 'dart:convert';
import 'dart:io';
import 'package:image/image.dart' as imglib;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:quiver/collection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  cameras = await availableCameras();

  runApp(
    MaterialApp(
      theme: ThemeData(primarySwatch: Colors.brown), 
      home: _MyHomePage(),
      title: "Face Recognition",
      debugShowCheckedModeBanner: false,
    ),
  );
}

class _MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<_MyHomePage> {
  File? jsonFile;
  Interpreter? interpreter;
  CameraController? camera;
  dynamic data = {};
  double threshold = 1.0;
  dynamic scanResults;
  CameraLensDirection direction = CameraLensDirection.front;
  late CameraDescription description = cameras[1];
  Directory? tempDir;
  bool faceFound = false;
  CameraImage? img;
  bool isBusy = false;
  dynamic faceDetector;
  List? e1;
  final TextEditingController name = TextEditingController();

  @override
  void initState() {
    super.initState();

    final options = FaceDetectorOptions(
        enableClassification: false,
        enableContours: false,
        enableLandmarks: false,
        performanceMode: FaceDetectorMode.fast);

    faceDetector = FaceDetector(options: options);

    cameraFunction();
  }

  Future loadModel() async {
    try {
      var interpreterOptions = InterpreterOptions();

      interpreter = await Interpreter.fromAsset('mobilefacenet.tflite',
          options: interpreterOptions);
    } on Exception {
      print('Failed to load model.');
    }
  }
////////////////////////////////////////////////////////////////////////////////////////////////

  // Future loadModel() async {
  //   try {
  //     final gpuDelegateV2 = GpuDelegateV2(
  //       options: GpuDelegateOptionsV2(
  //         isPrecisionLossAllowed: false,
  //         inferencePreference: TfLiteGpuInferenceUsage.fastSingleAnswer,
  //         inferencePriority1: TfLiteGpuInferencePriority.minLatency,
  //         inferencePriority2: TfLiteGpuInferencePriority.auto,
  //         inferencePriority3: TfLiteGpuInferencePriority.auto,
  //       ),
  //     );

  //     var interpreterOptions = InterpreterOptions()..addDelegate(gpuDelegateV2);
  //     interpreter = await tfl.Interpreter.fromAsset('mobilefacenet.tflite',
  //         options: interpreterOptions);
  //   } on Exception {
  //     print('Failed to load model.');
  //   }
  // }
////////////////////// Image Rotation //////////////////////////////////////////////////////////

  static InputImageRotation _getInputImageRotation(int sensorOrientation) {
    log("sensorOrientation: ${sensorOrientation}");
    if (sensorOrientation == 0) return InputImageRotation.rotation0deg;
    if (sensorOrientation == 90) return InputImageRotation.rotation90deg;
    if (sensorOrientation == 180) {
      return InputImageRotation.rotation180deg;
    } else {
      return InputImageRotation.rotation270deg;
    }
  }

  ////////////////camera method/////////////////////////////////////////////

  Future<void> cameraFunction() async {
    loadModel();
    camera =
        CameraController(description, ResolutionPreset.low, enableAudio: false);
    await camera!.initialize().then((_) {
      if (!mounted) {
        return;
      }
      camera!.startImageStream((image) => {
            if (!isBusy) {isBusy = true, img = image, doFaceDetectionOnFrame()}
          });
    });
  }

  //////////////// |_Face Detection Method_| ///////////////////////////////////////////

  doFaceDetectionOnFrame() async {
    tempDir = await getApplicationDocumentsDirectory();
    String embPath = '${tempDir!.path}/emb.json';
    jsonFile = File(embPath);
    if (jsonFile!.existsSync()) {
      data = json.decode(jsonFile!.readAsStringSync());
    }

    String resName;
    dynamic finalResult = Multimap<String, Face>();
    var frameImg = getInputImage();
    List<Face> faces = await faceDetector.processImage(frameImg);
    if (faces.isEmpty) {
      faceFound = false;
    } else {
      faceFound = true;
    }
    Face face;
    imglib.Image convertedImage = _convertCameraImage(img!, direction);
    for (face in faces) {
      var boundingBox1 = face.boundingBox;
      log(boundingBox1.toString());
      // final double? rotX = face.headEulerAngleX;
      // final double? rotY = face.headEulerAngleY;
      // final double? rotZ = face.headEulerAngleZ;
      double x, y, w, h;
      x = (face.boundingBox.left - 10);
      y = (face.boundingBox.top - 10);
      w = (face.boundingBox.width + 10);
      h = (face.boundingBox.height + 10);
      imglib.Image croppedImage = imglib.copyCrop(
        convertedImage,
        x.round(),
        y.round(),
        w.round(),
        h.round(),
      );
      croppedImage = imglib.copyResizeCropSquare(croppedImage, 112);
      resName = _recog(croppedImage);

      finalResult.add(resName, face);
    }
    setState(() {
      scanResults = finalResult;
      isBusy = false;
    });
  }

  ////////////////Face Detection Input//////////////////////////////////////////////////

  InputImage getInputImage() {
    final WriteBuffer allBytes = WriteBuffer();
    for (final Plane plane in img!.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();
    final Size imageSize = Size(
      img!.width.toDouble(),
      img!.height.toDouble(),
    );
    final camera = description;
    // final imageRotation =
    //     InputImageRotationValue.fromRawValue(camera.sensorOrientation);

    final imageRotation = _getInputImageRotation(camera.sensorOrientation);
    log('orientation: ${imageRotation.rawValue}');

    final inputImageFormat =
        InputImageFormatValue.fromRawValue(img!.format.raw);

    final planeData = img!.planes.map(
      (Plane plane) {
        return InputImagePlaneMetadata(
          bytesPerRow: plane.bytesPerRow,
          height: plane.height,
          width: plane.width,
        );
      },
    ).toList();

    final inputImageData = InputImageData(
      size: imageSize,
      imageRotation: imageRotation,
      inputImageFormat: inputImageFormat!,
      planeData: planeData,
    );

    final inputImage =
        InputImage.fromBytes(bytes: bytes, inputImageData: inputImageData);

    return inputImage;
  }

  ///////////////////////////////////////////////////////////////////////////////////////

  Widget _buildResults() {
    const Text noResultsText = Text('');
    if (scanResults == null || camera == null || !camera!.value.isInitialized) {
      return noResultsText;
    }
    CustomPainter painter;

    final Size imageSize = Size(
      camera!.value.previewSize!.height,
      camera!.value.previewSize!.width,
    );
    painter = FaceDetectorPainter(imageSize, scanResults, direction);
    return CustomPaint(
      painter: painter,
    );
  }

  Widget _buildImage() {
    if (camera == null || !camera!.value.isInitialized) {
      return Center(
        child: CircularProgressIndicator(),
      );
    }

    return Container(
      child: camera == null
          ? const Center(child: null)
          : Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(camera!),
                _buildResults(),
              ],
            ),
    );
  }
//////////////////////Function for changing camera direction//////////////////////////////////////

  void _toggleCameraDirection() async {
    if (direction == CameraLensDirection.back) {
      direction = CameraLensDirection.front;
      description = cameras[1];
    } else {
      direction = CameraLensDirection.back;
      description = cameras[0];
    }
    await camera!.stopImageStream();

    setState(() {
      // ignore: unnecessary_statements
      camera;
    });

    cameraFunction();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Center(
          child: const Text('Face recognition'),
        ),
        actions: [
          PopupMenuButton<Choice>(
            onSelected: (Choice result) {
              if (result == Choice.delete) {
                _resetFile();
              }
            },
            itemBuilder: (BuildContext context) => <PopupMenuEntry<Choice>>[
              const PopupMenuItem<Choice>(
                value: Choice.delete,
                child: Text('Remove all faces'),
              )
            ],
          ),
        ],
      ),
      body: _buildImage(),
      floatingActionButton:
          Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        FloatingActionButton(
          backgroundColor: (faceFound) ? Colors.brown : Colors.black,
          onPressed: () {
            if (faceFound) _addLabel();
          },
          heroTag: null,
          child: Icon(Icons.add),
        ),
        SizedBox(
          width: 10,
        ),
        FloatingActionButton(
          onPressed: _toggleCameraDirection,
          heroTag: null,
          child: Icon(Icons.camera_alt_rounded),
        ),
      ]),
    );
  }

  imglib.Image _convertCameraImage(CameraImage image, CameraLensDirection dir) {
    int width = image.width;
    int height = image.height;
    var img = imglib.Image(width, height);
    const int hexFF = 0xFF000000;
    final int uvyButtonStride = image.planes[1].bytesPerRow;
    final int? uvPixelStride = image.planes[1].bytesPerPixel;
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        final int uvIndex = uvPixelStride! * (x / 2).floor() +
            uvyButtonStride * (y / 2).floor();
        final int index = y * width + x;
        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
        img.data[index] = hexFF | (b << 16) | (g << 8) | r;
      }
    }
    var img1 = (dir == CameraLensDirection.front)
        ? imglib.copyRotate(img, -90)
        : imglib.copyRotate(img, 90);
    return img1;
  }

  String _recog(imglib.Image img) {
    List input = imageToByteListFloat32(img, 112, 128, 128);
    input = input.reshape([1, 112, 112, 3]);
    List output = List.filled(1 * 192, null, growable: false).reshape([1, 192]);
    interpreter?.run(input, output);
    output = output.reshape([192]);
    e1 = List.from(output);
    return compare(e1!).toUpperCase();
  }

  String compare(List currEmb) {
    if (data.length == 0) return "No Face saved";
    double minDist = 999;
    double currDist = 0.0;
    String predRes = "UNKNOWN";
    for (String label in data.keys) {
      currDist = euclideanDistance(data[label], currEmb);
      if (currDist <= threshold && currDist < minDist) {
        minDist = currDist;
        predRes = label;
      }
    }
    log(minDist.toString() + " " + predRes);
    return predRes;
  }

  void _resetFile() {
    data = {};
    jsonFile!.deleteSync();
  }

  void _addLabel() {
    setState(() {
      camera = null;
    });
    var alert = AlertDialog(
      title: Text("Add Face"),
      content: Row(
        children: [
          Expanded(
            child: TextField(
              controller: name,
              autofocus: true,
              decoration: InputDecoration(
                labelText: "Name",
                icon: Icon(Icons.face),
              ),
            ),
          )
        ],
      ),
      actions: [
        TextButton(
            child: Text("Save"),
            onPressed: () {
              _handle(
                name.text.toUpperCase(),
              );
              name.clear();
              Navigator.pop(context);
            }),
        TextButton(
          child: Text("Cancel"),
          onPressed: () {
            cameraFunction();
            Navigator.pop(context);
          },
        )
      ],
    );
    showDialog(
        context: context,
        builder: (context) {
          return alert;
        });
  }

  void _handle(String text) {
    data[text] = e1;
    jsonFile!.writeAsStringSync(
      json.encode(data),
    );
    cameraFunction();
  }
}

