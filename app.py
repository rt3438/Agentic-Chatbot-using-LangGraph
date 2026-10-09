from backend import (
    chatbot, 
    get_all_threads,
    ingest_rag_document
)

from langgraph.types import Command

from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    ToolMessage
)

import streamlit as st
import uuid
import tempfile
import os

# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    st.session_state["pending_interrupt"] = None
    
    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # return saved messages
    # return an empty list if no messages are available
    return state.values.get("messages", [])


def get_pending_interrupt(thread_id):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = chatbot.get_state(config)

    if state.interrupts:
        return state.interrupts[0]

    return None

def stream_graph(graph_input, config):

    interrupt_holder = {
        "interrupt": None
    }

    status_holder = {
        "box": None
    }

    def ai_only_stream():

        for mode, chunk in chatbot.stream(

            graph_input,

            config=config,

            stream_mode=[
                "messages",
                "updates"
            ]
        ):

            # ==================================
            # Graph updates /HITL
            # ==================================

            if mode == "updates":

                interrupts = chunk.get(
                    "__interrupt__",
                    ()
                )

                if interrupts:

                    interrupt_holder["interrupt"] = (
                        interrupts[0]
                    )

                continue

            # =================================
            # Message streaming
            # =================================

            if mode == "messages":

                message_chunk, metadata = chunk

                # Tool activity
                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = (
                            st.status(
                                f"🔧 Using `{tool_name}` ...",
                                expanded=True
                            )
                        )

                    else:

                        status_holder["box"].update(
                            label=(
                                f"🔧 Using "
                                f"`{tool_name}` ..."
                            ),
                            state = "running",
                            expanded=True
                        )

                # Assistant tokens
                if isinstance(
                    message_chunk,
                    AIMessage
                ):

                    if message_chunk.content:

                        yield (
                            message_chunk.content
                        )


    ai_message = st.write_stream(
        ai_only_stream()
    )


    # Finsh tool status only if graph
    # did NOT pause for approval
    if (
        status_holder["box"] is not None
        and interrupt_holder["interrupt"] is None
    ):

        status_holder["box"].update(
            label="✅ Tool finished",
            state="complete",
            expanded=True
        )

    return (
        ai_message,
        interrupt_holder["interrupt"]
    )



st.title("Agentic Chatbot with LangGraph")


# Create message_history when the app runs for the first time
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None


# Add the current thread to the conversation list if it's not already there
add_thread(st.session_state["thread_id"])

#======================================================== Sidebar threading feature ================================================

# Display the sidebar title
st.sidebar.title("My Conversations")


# Create a button for starting a new conversation
if st.sidebar.button("New Conversation"):

    # Reset the current chat and crate a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()


# Display all conversation threads in reverse order
# This shows the newest conversation first.
for thread_id in st.session_state["chat_threads"][::-1]:

    # Create a button for each conversation thread
    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):
        # Set the selected thread ID as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # Loop through all saved messages for the selected thread
        for message in messages:

            # Cheeck whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = 'user'

            # Check whether the message was sent by the assistant
            elif isinstance(message, AIMessage):
                # Skip empty tool-call AI messages
                if not message.content:
                    continue

                role = 'assistant'

            # Ignore other message types, such as ToolMessages
            else:
                continue

            # Convert the LangChain message into a dictionary
            temp_messages.append({
                'role': role,
                'content': message.content
            })

        # Replace the current UI hisory with the selected conversation's messages
        st.session_state['message_history'] = temp_messages

        # Restore pending HITL request if this
        # converstion was paused
        st.session_state['pending_interrupt'] = (
            get_pending_interrupt(thread_id)
        )

        # Rerun the Streamlit app to update the chat interface with the selected conversation
        st.rerun()
            

#======================================================== Main chat interface ======================================================

# Display all messages from the currently selected conversation
for message in st.session_state['message_history']:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message['role']):

        # Display the message content
        st.text(message['content'])

# HITL box will appear here
approval_placeholder = st.empty()

if st.session_state["pending_interrupt"]:

    pending = (
        st.session_state[
            "pending_interrupt"
        ]
    )

    st.warning(
        "⚠️ Human approval required"
    )

    with st.container(border=True):

        st.subheader(
            "🔧 Stock Purchase Approval"
        )

        st.write(
            pending.value
        )

        col1, col2 = st.columns(2)

        # ==========================
        # Approve
        # ==========================
        with col1:

            if st.button(
                "Approve",
                type="primary",
                use_container_width=True
            ):
                CONFIG = {

                    "configurable": {
                        "thread_id": 
                            st.session_state[
                                "thread_id"
                            ]
                    },

                    "metadata": {
                        "thread_id": 
                            st.session_state[
                                "thread_id"
                            ]
                    },

                    "run_name": "chat_trace"
                }

                resume_values = {
                    pending.id: "yes"
                }

                with st.chat_message(
                    "assistant"
                ):

                    ai_message, new_interrupt = (
                        stream_graph(
                            Command(
                                resume=resume_values
                            ),
                            CONFIG
                        )
                    )

                st.session_state[
                    "pending_interrupt"
                ] = new_interrupt


                if ai_message:

                    st.session_state[
                        "message_history"
                    ].append({
                        "role": 
                            "assistant",

                        "content": 
                            ai_message
                    })

                st.rerun()


        # =============================
        # Reject
        # =============================

        with col2:

            if st.button(
                "❌ Reject",
                use_container_width=True
            ):
                CONFIG = {

                    "configurable": {
                        "thread_id": 
                            st.session_state[
                                "thread_id"
                            ]
                    },

                    "metadata": {
                        "thread_id": 
                            st.session_state[
                                "thread_id"
                            ]
                    },

                    "run_name": 
                    "chat_trace"
                }

                resume_values = {
                    pending.id: "no"
                }

                with st.chat_message(
                    "assistant"
                ):

                    ai_message, new_interrupt = (
                        stream_graph(
                            Command(
                                resume=resume_values
                            ),
                            CONFIG
                        )
                    )

                st.session_state[
                    "pending_interrupt"
                ] = new_interrupt


                if ai_message:

                    st.session_state[
                        "message_history"
                    ].append({
                      
                        "role": 
                            "assistant",

                        "content": 
                            ai_message
                    })

                st.rerun()

# ======================================================= Fixed chat input with PDF upload ========================================


# keep st.chat_input directly in the main interface
# This keeps it fixed at the bottom
#
# accept_file=True adds the attachment button inside the chat input.
# file_type=["PDF"] allows PDF files only.
submission = None

if not st.session_state["pending_interrupt"]:

    submission = st.chat_input(
        "Ask a question:",
        accept_file=True,
        file_type=["PDF"]
    )

# Create the chat input box
# user_input = st.chat_input("Ask a question:")
user_input = None

# Process the submitted text and PDF
if submission:

    # Get the text entered b the user
    user_input = submission.text

    # Get the uploaded files
    # This is always a list when accept_file is enabled
    uploaded_files = submission.files

    # Process the uploaded PDF if one was attached
    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        # Store the temporary file path
        temporary_file_path = None

        try:

            # Save the uploaded PDF as a tempory local file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                # Write the uploaded PDF content to the temporary file
                temp_file.write(
                    uploaded_pdf.getvalue()
                )
                
                temporary_file_path = temp_file.name

            # Call the existing backend RAG ingestion function
            with st.spinner(
                f"Processing {uploaded_pdf.name}..."
            ):

                ingest_rag_document(
                    temporary_file_path
                )


            # Display PDF processing confirmation
            st.toast(
                f"{uploaded_pdf.name} processed successfully!",
                icon="✅"
            )

        except Exception as error:

            # Display PDF processing error
            st.error(
                f"Error processing PDF: {error}"
            )

        finally:

            # Delete the temporary file after indexing
            if (
                temporary_file_path
                and os.path.exists(temporary_file_path)
            ):
                os.remove(temporary_file_path)



# Run this block after the user submits a message
if user_input:

    # Save the user's message in Streamlit session state
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)



    CONFIG = {
        'configurable': {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }
  

    # Assistant streamnig block
    with st.chat_message("assistant"):

        ai_message, pending_interrupt = stream_graph(

            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ]
            },

            CONFIG
        )

    if pending_interrupt:

        st.session_state["pending_interrupt"] = (
            pending_interrupt
        )

        st.rerun()


    if ai_message:

        st.session_state[
            "message_history"
        ].append({

            "role": "assistant",

            "content": ai_message
        })

    # Save the complete assistant response in Streamlit session state
    # st.session_state['message_history'].append({
    #     'role': 'assistant',
    #     'content': ai_message
    # })