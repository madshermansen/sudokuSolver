import { StatusBar } from "expo-status-bar";
import { Platform, StyleSheet } from "react-native";

import EditScreenInfo from "@/components/EditScreenInfo";
import { Text, View } from "@/components/Themed";
import { useEffect, useState } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";

export default function Solve() {
  const [sudokuAnswer, setSudokuAnswer] = useState<number[][] | null>(null);
  const [solvedByModel, setSolvedByModel] = useState<boolean[][] | null>(null);

  useEffect(() => {
      const fetchData = async () => {
          const storedData = await AsyncStorage.getItem("solveData");
          if (storedData) {
            setSudokuAnswer(JSON.parse(storedData).sudokuAnswer);
            setSolvedByModel(JSON.parse(storedData).solvedByModel);
          }
      };

      fetchData();
  }, []);

  function createGrid(data: number[][] | null, solvedByModel: boolean[][] | null) {
    if (!data || !solvedByModel) {
        return null;
    }

    return data.map((row, rowIndex) => (
        <View key={rowIndex} style={styles.row}>
            {row.map((cell, cellIndex) => {
                const isSolvedByModel = solvedByModel[rowIndex]?.[cellIndex];
                return (
                    <Text
                        key={cellIndex}
                        style={[
                            styles.cell,
                            isSolvedByModel && styles.solvedCell, // Apply red style if solved by model
                        ]}
                    >
                        {cell}
                    </Text>
                );
            })}
        </View>
    ));
}

  

  return (
    <View style={styles.container}>
      <View style={styles.sudokuWrapper}>
      {createGrid(sudokuAnswer, solvedByModel)}
      </View>

      {/* Use a light status bar on iOS to account for the black space above the modal */}
      <StatusBar style={Platform.OS === "ios" ? "light" : "auto"} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
  },
  separator: {
    marginVertical: 30,
    height: 1,
    width: "80%",
  },
    row: {
        flexDirection: "row",
    },
    cell: {
        borderWidth: 1,
        width: 40,
        height: 40,
        textAlign: "center",
        lineHeight: 40,
        fontSize: 20,
    },
    sudokuWrapper: {
        borderWidth: 4,
    },
    solvedCell: {
        color: "red",
    },

});
