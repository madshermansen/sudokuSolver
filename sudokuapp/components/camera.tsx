import { CameraView, CameraType, useCameraPermissions } from "expo-camera";
import { useState } from "react";
import { Button, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { useRef } from "react";
import { Image } from "react-native";
import { useRouter } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";

export default function Camera() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const [photoUri, setPhotoUri] = useState<string | null>(null); // State to store the photo URI
  const router = useRouter();

  const serverURL =
    "http://172.27.31.20:8080/";
  const dir = "/api/solve";

  if (!permission) {
    // Camera permissions are still loading.
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View style={styles.container}>
        <Text style={styles.message}>
          We need your permission to show the camera
        </Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  async function takePicture() {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync();
        if (!photo) {
          console.error("No photo taken");
          return;
        }
        setPhotoUri(photo.uri); // Save the photo URI to state
        console.log("Photo taken:", photo.uri);
      } catch (error) {
        console.error("Error taking picture:", error);
      }
    }
  }

  function retakePicture() {
    setPhotoUri(null); // Clear the photo URI to show the camera again
  }

  async function sendImage() {
    if (!photoUri) {
      console.error("No photo to send");
      return;
    }

    try {
      // Prepare the form data
      const formData = new FormData();
      formData.append("image", {
        uri: photoUri, // The URI from CameraView
        type: "image/jpeg", // The MIME type of the file
        name: "photo.jpg", // The name of the file
      } as any); // Type assertion to bypass TypeScript errors

      // Send the image to the server
      const response = await fetch(`${serverURL}${dir}`, {
        method: "POST",
        body: formData,
        headers: {
          // Do NOT set Content-Type here; fetch will set it automatically.
        },
      });

      console.log("Server response status:", response.status);

      if (!response.ok) {
        const errorResponse = await response.text();
        throw new Error(`Server error: ${errorResponse}`);
      }

      // Parse and log the server response
      const result = await response.json();

      if (result.data) {
        await AsyncStorage.setItem("solveData", JSON.stringify(result.data));
        router.push("/solve");
      }
      console.log("Server response:", result);
    } catch (error) {
      console.error("Error sending image:", error);
    }
  }

  if (photoUri) {
    // If a photo has been taken, show it
    return (
      <View style={styles.container}>
        <Image source={{ uri: photoUri }} style={styles.preview} />
        <Button title="Retake" onPress={retakePicture} />
        <Button title="Send" onPress={sendImage} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        style={styles.camera}
        facing={'back'}
        ref={cameraRef}
        autofocus="off"
      >
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={takePicture}>
            <View style={styles.cameraButtonWrapper}>
            </View>
            <View style={styles.cameraButton}>
            </View>
          </TouchableOpacity>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
  },
  message: {
    textAlign: "center",
    paddingBottom: 10,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: "column",
    margin: 64,
    height: 100,
    bottom: 0,
  },
  button: {
    flex: 1,
    justifyContent: "center",
    height: "10%",
    alignItems: "center",
  },
  text: {
    fontSize: 24,
    fontWeight: "bold",
    color: "white",
  },
  preview: {
    flex: 1,
    width: "100%",
    resizeMode: "contain", // Ensure the image fits the screen
  },
  cameraButton: {
    position: "absolute",
    bottom: 0,
    width: 64,
    height: 64,
    borderRadius: 32,
    margin: 4,
    backgroundColor: "white",
  },
  cameraButtonWrapper: {
    position: "absolute",
    bottom: 0,
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 2,
    borderColor: "white",
  
  },
});
