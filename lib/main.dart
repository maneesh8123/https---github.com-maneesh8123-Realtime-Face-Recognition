import 'package:flutter/foundation.dart';
//import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import 'detector_painters.dart';
import 'utils.dart';

import 'dart:convert';
import 'dart:io';
// ignore: import_of_legacy_library_into_null_safe
import 'package:image/image.dart' as imglib;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:quiver/collection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
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
  CameraController? _camera;
  dynamic data = {};
  double threshold = 1.0;
  dynamic _scanResults;
  CameraLensDirection _direction = CameraLensDirection.front;
  late CameraDescription description = cameras[1];
  Directory? tempDir;
  bool _faceFound = false;
  CameraImage? img;
  bool isBusy = false;
  dynamic faceDetector;

  List? e1;
  bool loading = true;
  final TextEditingController _name = TextEditingController();

  @override
  void initState() {
    super.initState();
    final options = FaceDetectorOptions(
        enableClassification: false,
        enableContours: false,
        enableLandmarks: false,
        performanceMode: FaceDetectorMode.fast);

    //TODO initialize face detector
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

  // InputImageRotation rotationIntToImageRotation(int rotation) {
  //   switch (rotation) {
  //     case 0:
  //       return InputImageRotation.rotation0deg;
  //     case 90:
  //       return InputImageRotation.rotation90deg;
  //     case 180:
  //       return InputImageRotation.rotation180deg;
  //     default:
  //       assert(rotation == 270);
  //       return InputImageRotation.rotation270deg;
  //   }
  // }
  //   static InputImageRotation _getInputImageRotation(int sensorOrientation) {
  //   if (sensorOrientation == 0) return InputImageRotation.rotation0deg;
  //   if (sensorOrientation == 90) return InputImageRotation.rotation90deg;
  //   if (sensorOrientation == 180) return InputImageRotation.rotation180deg;
  //   return InputImageRotation.rotation270deg;
  // }

  ////////////////camera method////////////////////

  Future<void> cameraFunction() async {
    loadModel();
    _camera =
        CameraController(description, ResolutionPreset.low, enableAudio: false);
    await _camera!.initialize().then((_) {
      if (!mounted) {
        return;
      }
      _camera!.startImageStream((image) => {
            if (!isBusy) {isBusy = true, img = image, doFaceDetectionOnFrame()}
          });
    });
  }

  // @override
  // void dispose() {
  //   _camera!.dispose();
  //   faceDetector.close();
  //   super.dispose();
  // }
  ////////////////Face Detection Method////////////////////

  doFaceDetectionOnFrame() async {
    tempDir = await getApplicationDocumentsDirectory();
    String embPath = '${tempDir!.path}/emb.json';
    jsonFile = File(embPath);
    if (jsonFile!.existsSync()) {
      data = json.decode(jsonFile!.readAsStringSync());
    }

    String res_name;
    dynamic finalResult = Multimap<String, Face>();
    var frameImg = getInputImage();
    List<Face> faces = await faceDetector.processImage(frameImg);
    if (faces.isEmpty) {
      _faceFound = false;
    } else {
      _faceFound = true;
    }
    Face face;
    imglib.Image convertedImage = _convertCameraImage(img!, _direction);
    for (face in faces) {
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
      res_name = _recog(croppedImage);

      finalResult.add(res_name, face);
    }
    setState(() {
      _scanResults = finalResult;
      isBusy = false;
    });
  }

  ////////////////Face Detection/////////////////////////

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
    final imageRotation =
        InputImageRotationValue.fromRawValue(camera.sensorOrientation);

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
      imageRotation: imageRotation!,
      inputImageFormat: inputImageFormat!,
      planeData: planeData,
    );

    final inputImage =
        InputImage.fromBytes(bytes: bytes, inputImageData: inputImageData);

    return inputImage;
  }

  ///////////////////////////////////////////////////////////////////////////////

  Widget _buildResults() {
    const Text noResultsText = Text('');
    if (_scanResults == null ||
        _camera == null ||
        !_camera!.value.isInitialized) {
      return noResultsText;
    }
    CustomPainter painter;

    final Size imageSize = Size(
      _camera!.value.previewSize!.height,
      _camera!.value.previewSize!.width,
    );
    painter = FaceDetectorPainter(imageSize, _scanResults,_direction);
    return CustomPaint(
      painter: painter,
    );
  }

  Widget _buildImage() {
    if (_camera == null || !_camera!.value.isInitialized) {
      return Center(
        child: CircularProgressIndicator(),
      );
    }

    return Container(
      constraints: const BoxConstraints.expand(),
      child: _camera == null
          ? const Center(child: null)
          : Stack(
              fit: StackFit.expand,
              children: <Widget>[
                CameraPreview(_camera!),
                _buildResults(),
              ],
            ),
    );
  }
//////////////////////Function for changing camera direction//////////////////////////////////////

  void _toggleCameraDirection() async {
    if (_direction == CameraLensDirection.back) {
      _direction = CameraLensDirection.front;
      description = cameras[1];
    } else {
      _direction = CameraLensDirection.back;
      description = cameras[0];
    }
    await _camera!.stopImageStream();

    setState(() {
      // ignore: unnecessary_statements
      _camera;
    });

    cameraFunction();
  }

  @override
  Widget build(BuildContext context) {
    //List<Widget> stackChildren = [];
    return Scaffold(
      appBar: AppBar(
        title: Center(
          child: const Text('Face recognition'),
        ),
        actions: <Widget>[
          PopupMenuButton<Choice>(
            onSelected: (Choice result) {
              if (result == Choice.delete) {
                _resetFile();
              } else {
                _viewLabels();
              }
            },
            itemBuilder: (BuildContext context) => <PopupMenuEntry<Choice>>[
              const PopupMenuItem<Choice>(
                value: Choice.view,
                child: Text('View Saved Faces'),
              ),
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
          backgroundColor: (_faceFound) ? Colors.blue : Colors.blueGrey,
          onPressed: () {
            if (_faceFound) _addLabel();
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

  imglib.Image _convertCameraImage(
      CameraImage image, CameraLensDirection _dir) {
    int width = image.width;
    int height = image.height;
    // imglib -> Image package from https://pub.dartlang.org/packages/image
    var img = imglib.Image(width, height); // Create Image buffer
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
        // Calculate pixel color
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
        // color: 0x FF  FF  FF  FF
        //           A   B   G   R
        img.data[index] = hexFF | (b << 16) | (g << 8) | r;
      }
    }
    var img1 = (_dir == CameraLensDirection.front)
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
    print(minDist.toString() + " " + predRes);
    return predRes;
  }

  void _resetFile() {
    data = {};
    jsonFile!.deleteSync();
  }

  void _viewLabels() {
    setState(() {
      _camera = null;
    });
    String name;
    var alert = AlertDialog(
      title: Text("Saved Faces"),
      content: ListView.builder(
          padding: EdgeInsets.all(2),
          itemCount: data.length,
          itemBuilder: (BuildContext context, int index) {
            name = data.keys.elementAt(index);
            return Column(
              children: <Widget>[
                ListTile(
                  title: Text(
                    name,
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.grey[400],
                    ),
                  ),
                ),
                Padding(
                  padding: EdgeInsets.all(2),
                ),
                Divider(),
              ],
            );
          }),
      actions: <Widget>[
        TextButton(
          child: Text("OK"),
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

  void _addLabel() {
    setState(() {
      _camera = null;
    });
    print("Adding new face");
    var alert = AlertDialog(
      title: Text("Add Face"),
      content: Row(
        children: <Widget>[
          Expanded(
            child: TextField(
              controller: _name,
              autofocus: true,
              decoration: InputDecoration(
                labelText: "Name",
                icon: Icon(Icons.face),
              ),
            ),
          )
        ],
      ),
      actions: <Widget>[
        TextButton(
            child: Text("Save"),
            onPressed: () {
              _handle(
                _name.text.toUpperCase(),
              );
              _name.clear();
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
