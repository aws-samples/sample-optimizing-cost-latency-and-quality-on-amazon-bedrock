"""Synthetic headset conversation and local check references for the context lab.

Assistant messages are supplied examples, not recorded model outputs.
Required evidence is used for local validation, never added to model prompts.
"""

from __future__ import annotations

from .context_management import Exchange

SUPPORT_EXCHANGES = [
    Exchange(
        "E1",
        "Hi, the left side of my H-200 headset has stopped working. Same problem with music and calls. "
        "It's a work headset, so I can't install software or do a factory reset.",
        "Let's start with checks that won't change its settings. Have you tried a different cable "
        "or plugged it into another device?",
    ),
    Exchange(
        "E2",
        "I tried a cable from a working headset, but the left side is still silent. "
        "I haven't run the hardware diagnostic yet. IT hasn't approved a replacement.",
        "Thanks for trying that. Could you check the model on the headset's label "
        "before we choose the diagnostic steps?",
    ),
    Exchange(
        "E3",
        "Ah, it's H-220, not H-200. I was reading the old box! I'll try it with my tablet now.",
        "Got it—H-220. Please play the same recording on the tablet "
        "and let me know whether the left side works.",
    ),
    Exchange(
        "E4",
        "Still no sound on the left when I use the tablet. The tablet's own speakers work fine.",
        "I'll get the H-220 hardware diagnostic steps and check that they're suitable "
        "for your work-device restrictions. Would you prefer written instructions?",
    ),
    Exchange(
        "E5",
        "Yes, written steps please. I have a meeting in five minutes, so I'll need to come back to this.",
        "No problem. I'll put the instructions in the case rather than asking you to remember them. "
        "When you return, use this chat so the technician can see the cable and tablet checks. "
        "There's no need to repeat those while you're waiting.",
    ),
    Exchange(
        "E6",
        "Back now. Tomorrow morning would be better for the diagnostic. "
        "I haven't tried anything else since the tablet test.",
        "Tomorrow is fine. Leave the headset as it is for now. "
        "I'll ask the technician for written instructions that work within your IT restrictions. "
        "If the suggested procedure needs an installation or reset, we'll need a different route.",
    ),
    Exchange(
        "E7",
        "One other thing: I found order B-67890 while looking for the receipt. "
        "That's for my keyboard, though.",
        "Thanks—I'll keep that separate from the headset case. Do you have the headset receipt? "
        "A PDF or a clear photo is fine. It helps the team check the purchase details "
        "if the case later needs a repair or replacement review.",
    ),
    Exchange(
        "E8",
        "Yes, I've found the receipt PDF. It has my home address on it. Can I cover that before sending it?",
        "Yes, cover the home address before sharing it here. Leave the purchase date, retailer, "
        "item description and order reference visible. Please don't include payment-card details. "
        "The team can ask for anything else it needs through the approved support channel.",
    ),
    Exchange(
        "E9",
        "The item description says H-200 on the receipt too. But the label on the headset definitely says H-220.",
        "Thanks for flagging the mismatch. We'll use the headset label for troubleshooting and "
        "note the receipt discrepancy separately for the purchase review. "
        "Please include a photo of the label with the receipt so the team can follow up with the retailer.",
    ),
    Exchange(
        "E10",
        "I've uploaded both. The link for the instructions asks me to sign in, and my work account doesn't open it.",
        "Sorry about that. I'll request a plain-text copy in this case. "
        "You don't need to create another account or install anything to read it. "
        "Please wait for the accessible instructions before starting the diagnostic.",
    ),
    Exchange(
        "E11",
        "Should I pack the headset for collection? I'm working from home this week.",
        "Not yet. We still need the diagnostic and the team's decision on the case. "
        "Don't send the headset until you receive collection instructions. "
        "Keep the packaging nearby, and we'll confirm where to send it if a return is arranged.",
    ),
    Exchange(
        "E12",
        "Could someone contact me between nine and ten tomorrow? Email is better than a call.",
        "I'll add that preference to the case. The technician will confirm availability by email. "
        "If that time doesn't work, you can reply with another window without opening a new support request.",
    ),
    Exchange(
        "E13",
        "Will it be you helping tomorrow, or someone else? I don't want to start the whole explanation again.",
        "It may be another technician. They'll have this conversation, your attachments and the case notes. "
        "I'll make sure the handoff separates what you've already tried from the diagnostic that's still pending.",
    ),
    Exchange(
        "E14",
        "Great, thank you. I'm signing off now. The headset is still on my desk.",
        "Thanks for your patience. We'll continue in this case when the written instructions are ready. "
        "Please keep an eye on your email for the appointment confirmation.",
    ),
]

REQUIRED_EVIDENCE = {
    "E1": ["It's a work headset, so I can't install software or do a factory reset."],
    "E2": [
        "I tried a cable from a working headset, but the left side is still silent.",
        "I haven't run the hardware diagnostic yet.",
        "IT hasn't approved a replacement.",
    ],
    "E3": ["Ah, it's H-220, not H-200."],
    "E4": ["Still no sound on the left when I use the tablet."],
}

CHECKPOINTS = {
    4: "Prepare the support handoff from the evidence so far.",
    6: "Refresh the handoff after the customer returns. What is still unresolved?",
    8: "Update the technician's handoff after the receipt discussion.",
    10: "Update the handoff after the attachment and instructions-access discussion.",
    12: "Refresh the handoff for the technician handling tomorrow's appointment.",
    14: "Give the final handoff, preserving corrections and avoiding repeated failed checks.",
}
