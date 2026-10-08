import {
  ConsoleTemplate,
  FullScreenContainer,
} from "@pipecat-ai/voice-ui-kit";


export default function App() {
  return (
    <FullScreenContainer>
      <ConsoleTemplate
        titleText="Multilingual Voice AI"
        transportType="smallwebrtc"
        connectParams={{
          webrtcUrl: "/offer",
        }}
        transportOptions={{
          waitForICEGathering: true,
          iceServers: [
            {
              urls: "stun:stun.l.google.com:19302",
            },
          ],
        }}
        noScreenControl
        noUserVideo
        assistantLabelText="assistant"
      />
    </FullScreenContainer>
  );
}