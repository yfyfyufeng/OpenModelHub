package main

import (
	"encoding/json"
	"io"
	"strings"

	"errors"

	userlib "github.com/cs161-staff/project2-userlib"
	"github.com/google/uuid"

	"bufio"
	"fmt"
	"net"
	_ "strconv"
)

type UUID = userlib.UUID

type User struct {
	Username   string
	ID         UUID
	Password   string
	FileAccess []FilenameFileIDKey
	// Invitations    []UUID // invitations this user send out
	EncPublicKey   userlib.PKEEncKey
	DecPrivateKey  userlib.PKEDecKey
	SignPrivateKey userlib.DSSignKey
	VerPublicKey   userlib.DSVerifyKey
	SymKey         []byte
}
type FilenameFileIDKey struct {
	//ThisID          UUID
	Filename        string
	FileID          UUID
	InvitationKey   []byte
	FileOwnerVerKey userlib.DSVerifyKey
	KeyManagerID    UUID
}
type UserPubKey struct {
	Username     string
	EncPublicKey userlib.PKEEncKey
	VerPublicKey userlib.DSVerifyKey
}

type File struct {
	ID              UUID
	EncFileContent  UUID // ptr to the file content
	EncInvKeys      UUID
	FileKeyManagers []UUID
}
type InvKeys struct {
	ID               UUID
	UsernameToInvKey [][]byte
}
type UsernameInvKey struct {
	SourceUsername string
	DestUsername   string
	InvitationID   UUID
	InvitationKey  []byte
}

type FileContent struct {
	FileContentID UUID
	FirstFileNode UUID
	LastFileNode  UUID
}
type FileNode struct {
	Content    []byte
	ThisNodeID UUID
	NextNodeID UUID
}

type KeyManager struct {
	KeyManagerID  UUID
	FileContentID UUID
	ContentKey    []byte
}

type Invitation struct {
	ID              UUID
	SourceUser      string
	DestUser        string
	FileOwnerVerKey userlib.DSVerifyKey
	File            UUID
	Key             []byte
	KeyManagerID    UUID
}

// NOTE: The following methods have toy (insecure!) implementations.

func InitUser(username string, password string) (userdataptr *User, err error) {

	// Check username duplication
	if UserExist(username) {
		return nil, errors.New("username have been used")
	}

	// Check empty username
	if username == "" {
		return nil, errors.New("empty username not allowed")
	}

	var userdata User
	userdata.Username = username

	// Generate UUID
	hashedCredentials := userlib.Argon2Key([]byte(username+password), []byte("salt"), 16)
	userdata.ID, err = uuid.FromBytes(hashedCredentials)
	if err != nil {
		return nil, err
	}

	// Store pwd
	userdata.Password = password // userlib.Argon2Key([]byte(password), []byte("salt"), 16)

	// Init other fields
	userdata.FileAccess = []FilenameFileIDKey{}
	//userdata.Invitations = []UUID{}

	// Generate public/private keys
	userdata.EncPublicKey, userdata.DecPrivateKey, err = userlib.PKEKeyGen()
	if err != nil {
		return nil, err
	}
	userdata.SignPrivateKey, userdata.VerPublicKey, err = userlib.DSKeyGen()
	if err != nil {
		return nil, err
	}
	userdata.SymKey = userlib.Argon2Key([]byte(username+password), []byte("pwdsalt"), 16)

	// Init Keystore
	userlib.KeystoreSet(username+"Enc", userdata.EncPublicKey)
	userlib.KeystoreSet(username+"Ver", userdata.VerPublicKey)

	err = uploadUser(&userdata)
	if err != nil {
		return nil, err
	}

	return &userdata, nil
}

func GetUser(username string, password string) (userdataptr *User, err error) {
	// var userdata User
	// userdataptr = &userdata
	// return userdataptr, nil

	encOnlineUserData, found := Login(username, password)
	if !found {
		return nil, errors.New("user not found")
	}

	userSymKey := userlib.Argon2Key([]byte(username+password), []byte("pwdsalt"), 16)

	userDataBytes, err := SymAuthDec(userSymKey, encOnlineUserData)
	if err != nil {
		return nil, err
	}

	var userData User
	err = json.Unmarshal(userDataBytes, &userData)
	if err != nil {
		return nil, err
	}

	// Check username
	if userData.Username != username {
		return nil, errors.New("username does not match")
	}
	// Check pwd
	if !userlib.HMACEqual([]byte(userData.Password), []byte(password)) {
		return nil, errors.New("password does not match")
	}
	// Check symmetric key
	if !userlib.HMACEqual(userSymKey, userData.SymKey) {
		return nil, errors.New("sym key does not match")
	}

	return &userData, nil
}

func (userdata *User) StoreFile(filename string, content []byte) (err error) {
	err = downloadUser(userdata)
	if err != nil {
		return err
	}

	// Check if we are overwriting the file
	_, userFileAccess, err := UserGetFileAccess(userdata, filename)
	if err != nil {
		return err
	}

	isOverwriting := userFileAccess != nil

	if !isOverwriting { // Create a new file
		// Store File Content
		fileContentKey := userlib.RandomBytes(16)
		specialKey := userlib.RandomBytes(16)

		firstNodeID := uuid.New()
		secondNodeID := uuid.New()

		firstFileNode := FileNode{
			Content:    content,
			ThisNodeID: firstNodeID,
			NextNodeID: secondNodeID,
		}
		firstFileNodeBytes, err := json.Marshal(firstFileNode)
		if err != nil {
			return err
		}
		encFirstFileNodeBytes, err := SymAuthEnc(fileContentKey, firstFileNodeBytes)
		if err != nil {
			return err
		}
		userlib.DatastoreSet(firstNodeID, encFirstFileNodeBytes)

		secondFileNode := FileNode{
			Content:    nil,
			ThisNodeID: secondNodeID,
			NextNodeID: uuid.Nil,
		}
		secondFileNodeBytes, err := json.Marshal(secondFileNode)
		if err != nil {
			return err
		}
		encSecondFileNodeBytes, err := SymAuthEnc(fileContentKey, secondFileNodeBytes)
		if err != nil {
			return err
		}
		userlib.DatastoreSet(secondNodeID, encSecondFileNodeBytes)

		//Create File Content Struct
		fileContentID := uuid.New()
		fileContent := FileContent{
			FileContentID: fileContentID,
			//FileLength:    uint64(len(content)),
			FirstFileNode: firstNodeID,
			LastFileNode:  secondNodeID,
		}
		fileContentBytes, err := json.Marshal(fileContent)
		if err != nil {
			return err
		}
		encFileContentBytes, err := SymAuthEnc(fileContentKey, fileContentBytes)
		if err != nil {
			return err
		}
		userlib.DatastoreSet(fileContentID, encFileContentBytes)

		// Create file inv key struct
		invKeys := InvKeys{
			ID:               uuid.New(),
			UsernameToInvKey: [][]byte{},
		}
		invKeysBytes, err := json.Marshal(invKeys)
		if err != nil {
			return err
		}
		encInvKeysBytes, err := SymAuthEnc(fileContentKey, invKeysBytes)
		if err != nil {
			return err
		}
		userlib.DatastoreSet(invKeys.ID, encInvKeysBytes)

		// Create File Struct
		fileIDBytes := userlib.Argon2Key([]byte(filename), []byte("salt"), 16)
		fileID, err := uuid.FromBytes(fileIDBytes)
		if err != nil {
			return err
		}
		file := File{
			ID: fileID,
			// OwnerName:        userdata.Username,
			// OwnerNameSig:     signedOwnerName,
			// OriginalFilename: filename,
			EncFileContent: fileContentID,
			//EncFileSecretData: fileSecretDataID,
			FileKeyManagers: []UUID{},
			EncInvKeys:      invKeys.ID,
			//EncRelationships:  []UUID{},
		}
		fileBytes, err := json.Marshal(file)
		if err != nil {
			return err
		}
		userlib.DatastoreSet(fileID, fileBytes)

		// Manually add the user to the file manager
		filenameFileIDKey := FilenameFileIDKey{
			//ThisID:          uuid.New(),
			Filename:        filename,
			FileID:          fileID,
			InvitationKey:   specialKey,
			FileOwnerVerKey: userdata.VerPublicKey,
			KeyManagerID:    uuid.New(),
		}

		userdata.FileAccess = append(userdata.FileAccess, filenameFileIDKey)

		// Add the user to the file key manager
		keyManager := KeyManager{
			KeyManagerID:  filenameFileIDKey.KeyManagerID,
			FileContentID: fileContentID,
			ContentKey:    fileContentKey,
		}
		keyManagerBytes, err := json.Marshal(keyManager)
		if err != nil {
			return err
		}
		encKeyManagerBytes, err := SymAuthEnc(specialKey, keyManagerBytes)
		if err != nil {
			return err
		}
		userlib.DatastoreSet(keyManager.KeyManagerID, encKeyManagerBytes)
		file.FileKeyManagers = append(file.FileKeyManagers, keyManager.KeyManagerID)

		// Update file struct with the new key manager
		fileBytes, err = json.Marshal(file)
		if err != nil {
			return err
		}
		userlib.DatastoreSet(fileID, fileBytes)

	} else { // Overwrite the content
		// Get File Content Key
		fileContentKey, err := UserGetFileContentKey(userdata, filename)
		if err != nil {
			return err
		}

		// Store File Content
		firstNodeID := uuid.New()
		secondNodeID := uuid.New()

		firstFileNode := FileNode{
			Content:    content,
			ThisNodeID: firstNodeID,
			NextNodeID: secondNodeID,
		}
		firstFileNodeBytes, err := json.Marshal(firstFileNode)
		if err != nil {
			return err
		}
		encFirstFileNodeBytes, err := SymAuthEnc(fileContentKey, firstFileNodeBytes)
		if err != nil {
			return err
		}
		userlib.DatastoreSet(firstNodeID, encFirstFileNodeBytes)

		secondFileNode := FileNode{
			Content:    nil,
			ThisNodeID: secondNodeID,
			NextNodeID: uuid.Nil,
		}
		secondFileNodeBytes, err := json.Marshal(secondFileNode)
		if err != nil {
			return err
		}
		encSecondFileNodeBytes, err := SymAuthEnc(fileContentKey, secondFileNodeBytes)
		if err != nil {
			return err
		}
		userlib.DatastoreSet(secondNodeID, encSecondFileNodeBytes)

		// Get File Content
		fileContent, err := GetFile(userdata, filename)
		if err != nil {
			return err
		}
		fileContent.FirstFileNode = firstNodeID
		fileContent.LastFileNode = secondNodeID
		//fileContent.FileLength = uint64(len(content))

		fileContentBytes, err := json.Marshal(fileContent)
		if err != nil {
			return err
		}
		encFileContentBytes, err := SymAuthEnc(fileContentKey, fileContentBytes)
		if err != nil {
			return err
		}
		userlib.DatastoreSet(fileContent.FileContentID, encFileContentBytes)
	}

	err = uploadUser(userdata)
	if err != nil {
		return err
	}

	return nil
}

func (userdata *User) AppendToFile(filename string, content []byte) error {
	// err := downloadUser(userdata)
	// if err != nil {
	// 	return err
	// }

	// Get the file content structure using the helper function
	fileContent, err := GetFile(userdata, filename)
	if err != nil {
		return err
	}

	// Get File Content Key
	fileContentKey, err := UserGetFileContentKey(userdata, filename)
	if err != nil {
		return err
	}

	// Store content on the last node
	lastFileNodeID := fileContent.LastFileNode
	encLastNodeBytes, ok := userlib.DatastoreGet(lastFileNodeID)
	if !ok {
		return errors.New("last node not found")
	}
	lastNodeBytes, err := SymAuthDec(fileContentKey, encLastNodeBytes)
	if err != nil {
		return err
	}
	var lastNode FileNode
	err = json.Unmarshal(lastNodeBytes, &lastNode)
	if err != nil {
		return err
	}
	lastNode.Content = content

	// Create a new empty file node
	newNodeID := uuid.New()
	newNode := FileNode{
		Content:    nil,
		ThisNodeID: newNodeID,
		NextNodeID: uuid.Nil, // No next node initially
	}

	// Link nodes
	lastNode.NextNodeID = newNodeID
	fileContent.LastFileNode = newNodeID

	// Encrypt and store last node
	lastNodeBytes, err = json.Marshal(lastNode)
	if err != nil {
		return err
	}
	encLastNodeBytes, err = SymAuthEnc(fileContentKey, lastNodeBytes)
	if err != nil {
		return err
	}
	userlib.DatastoreSet(lastFileNodeID, encLastNodeBytes)

	// Encrypt and store new node
	newNodeBytes, err := json.Marshal(newNode)
	if err != nil {
		return err
	}
	encNewNodeBytes, err := SymAuthEnc(fileContentKey, newNodeBytes)
	if err != nil {
		return err
	}
	userlib.DatastoreSet(newNodeID, encNewNodeBytes)

	// Encrypt and store file content
	fileContentBytes, err := json.Marshal(fileContent)
	if err != nil {
		return err
	}
	encFileContentBytes, err := SymAuthEnc(fileContentKey, fileContentBytes)
	if err != nil {
		return err
	}
	userlib.DatastoreSet(fileContent.FileContentID, encFileContentBytes)

	return nil
}

func (userdata *User) LoadFile(filename string) (content []byte, err error) {
	err = downloadUser(userdata)
	if err != nil {
		return nil, err
	}

	// Get File Content UUID and Content Key
	fileContentKey, err := UserGetFileContentKey(userdata, filename)
	if err != nil {
		return nil, err
	}
	if fileContentKey == nil {
		return nil, errors.New("file accesss denied")
	}

	// Get file content struct
	fileContent, err := GetFile(userdata, filename)
	if err != nil {
		return nil, err
	}

	// Get content
	content = []byte{}
	currentNodeID := fileContent.FirstFileNode
	for {
		// Get current node
		encCurrentNodeBytes, ok := userlib.DatastoreGet(currentNodeID)
		if !ok {
			return nil, errors.New("file node not found")
		}

		currentNodeBytes, err := SymAuthDec(fileContentKey, encCurrentNodeBytes)
		if err != nil {
			return nil, err
		}

		var currentNode FileNode
		err = json.Unmarshal(currentNodeBytes, &currentNode)
		if err != nil {
			return nil, err
		}

		// Check end
		if currentNode.Content == nil && currentNode.NextNodeID == uuid.Nil {
			break
		}

		// Append content
		content = append(content, currentNode.Content...)

		// Go to next node
		currentNodeID = currentNode.NextNodeID
	}

	err = uploadUser(userdata)
	if err != nil {
		return nil, err
	}

	return content, err
}

func (userdata *User) CreateInvitation(filename string, recipientUsername string) (invitationPtr uuid.UUID, err error) {
	// Check if dest user exist
	if !UserExist(recipientUsername) {
		return uuid.Nil, errors.New("dest user not found")
	}

	err = downloadUser(userdata)
	if err != nil {
		return uuid.Nil, err
	}

	// Check if the user has access to the file
	_, hasAccess, err := UserGetFileAccess(userdata, filename)
	if err != nil {
		return uuid.Nil, err
	}
	if hasAccess == nil {
		return uuid.Nil, errors.New("user does not have access to the file")
	}

	// Check if recipient user exists
	if !UserExist(recipientUsername) {
		return uuid.Nil, errors.New("recipient user does not exist")
	}

	// Generate a random special key for the invitation
	specialKey := userlib.RandomBytes(16)

	// Get File Owner Public Ver Key
	var fileOwnerVerKey userlib.DSVerifyKey
	fileID := uuid.Nil
	flag := false
	for _, filenameFileIDKey := range userdata.FileAccess {
		if filenameFileIDKey.Filename == filename {
			flag = true
			fileID = filenameFileIDKey.FileID
			fileOwnerVerKey = filenameFileIDKey.FileOwnerVerKey
		}
	}

	if !flag {
		return uuid.Nil, errors.New("file access denied")
	}

	// Create the invitation struct
	invitation := Invitation{
		ID:              uuid.New(),
		SourceUser:      userdata.Username,
		DestUser:        recipientUsername,
		File:            fileID,
		Key:             specialKey,
		FileOwnerVerKey: fileOwnerVerKey,
		KeyManagerID:    uuid.New(),
	}

	// Serialize the invitation and store it in the datastore
	recipientPublicKey, _, err := GetUserPubKey(recipientUsername)
	if err != nil {
		return uuid.Nil, err
	}

	invitationBytes, err := json.Marshal(invitation)
	if err != nil {
		return uuid.Nil, err
	}

	encInvitationBytes, err := HybridEnc(recipientPublicKey, userdata.SignPrivateKey, invitationBytes)
	if err != nil {
		return uuid.Nil, err
	}

	userlib.DatastoreSet(invitation.ID, encInvitationBytes)

	// Append the invitation UUID to user's Invitations
	//userdata.Invitations = append(userdata.Invitations, invitation.ID)

	// Get file content key
	fileContentKey, err := UserGetFileContentKey(userdata, filename)
	if err != nil {
		return uuid.Nil, err
	}

	// Update the file's invitations keys to owner
	var fileStruct File
	fileData, ok := userlib.DatastoreGet(fileID)
	if !ok {
		return uuid.Nil, errors.New("file not found in datastore")
	}

	err = json.Unmarshal(fileData, &fileStruct)
	if err != nil {
		return uuid.Nil, err
	}

	encInvKeysBytes, ok := userlib.DatastoreGet(fileStruct.EncInvKeys)
	if !ok {
		return uuid.Nil, errors.New("Invitation Key not found")
	}

	invKeysBytes, err := SymAuthDec(fileContentKey, encInvKeysBytes)
	if err != nil {
		return uuid.Nil, err
	}

	var invKeys InvKeys
	err = json.Unmarshal(invKeysBytes, &invKeys)
	if err != nil {
		return uuid.Nil, err
	}

	usernameToKey := UsernameInvKey{
		SourceUsername: userdata.Username,
		DestUsername:   recipientUsername,
		InvitationKey:  specialKey,
		InvitationID:   invitation.ID,
	}

	usernameToKeyBytes, err := json.Marshal(usernameToKey)
	if err != nil {
		return uuid.Nil, err
	}

	// Encrypt usernameToKey and append to invKeys
	invKeys.UsernameToInvKey = append(invKeys.UsernameToInvKey, usernameToKeyBytes)

	// Update invKey
	invKeysBytes, err = json.Marshal(invKeys)
	if err != nil {
		return uuid.Nil, err
	}

	encInvKeysBytes, err = SymAuthEnc(fileContentKey, invKeysBytes)
	if err != nil {
		return uuid.Nil, err
	}

	userlib.DatastoreSet(invKeys.ID, encInvKeysBytes)

	// key manager op

	keyManager := KeyManager{
		KeyManagerID:  invitation.KeyManagerID,
		FileContentID: fileStruct.EncFileContent,
		ContentKey:    fileContentKey,
	}

	keyManagerBytes, err := json.Marshal(keyManager)
	if err != nil {
		return uuid.Nil, err
	}

	encKeyManagerBytes, err := SymAuthEnc(specialKey, keyManagerBytes)
	if err != nil {
		return uuid.Nil, err
	}

	userlib.DatastoreSet(keyManager.KeyManagerID, encKeyManagerBytes)

	err = uploadUser(userdata)
	if err != nil {
		return uuid.Nil, err
	}

	fileStruct.FileKeyManagers = append(fileStruct.FileKeyManagers, keyManager.KeyManagerID)

	// Update file struct with the new key manager
	fileBytes, err := json.Marshal(fileStruct)
	if err != nil {
		return uuid.Nil, err
	}
	userlib.DatastoreSet(fileID, fileBytes)

	// Return the invitation UUID
	return invitation.ID, nil
}

func (userdata *User) AcceptInvitation(senderUsername string, invitationPtr uuid.UUID, filename string) error {
	err := downloadUser(userdata)
	if err != nil {
		return err
	}

	// Check sender exist
	if !UserExist(senderUsername) {
		return errors.New("sender not found")
	}

	// Check duplicated filename
	_, fileExist, err := UserGetFileAccess(userdata, filename)
	if err != nil {
		return err
	}
	if fileExist != nil {
		return errors.New("filename duplicated")
	}

	// Get invitation
	encInvitationBytes, ok := userlib.DatastoreGet(invitationPtr)
	if !ok {
		return errors.New("invitation not found")
	}
	_, verifyKey, err := GetUserPubKey(senderUsername)
	if err != nil {
		return err
	}
	invitationBytes, err := HybridDec(userdata.DecPrivateKey, verifyKey, encInvitationBytes)
	if err != nil {
		return err
	}
	var invitation Invitation
	err = json.Unmarshal(invitationBytes, &invitation)
	if err != nil {
		return err
	}

	// Check invitation
	if senderUsername != invitation.SourceUser {
		return errors.New("sender username does not match with inviitation")
	}
	if userdata.Username != invitation.DestUser {
		return errors.New("receiver username does not match with inviitation")
	}

	// Store filename file UUID Key
	filenameFileIDKey := FilenameFileIDKey{
		Filename:        filename,
		FileID:          invitation.File,
		InvitationKey:   invitation.Key,
		FileOwnerVerKey: invitation.FileOwnerVerKey,
		KeyManagerID:    invitation.KeyManagerID,
	}
	userdata.FileAccess = append(userdata.FileAccess, filenameFileIDKey)

	err = uploadUser(userdata)
	if err != nil {
		return err
	}

	return nil
}

func (userdata *User) RevokeAccess(filename string, recipientUsername string) error {
	err := downloadUser(userdata)
	if err != nil {
		return err
	}

	// Check owner identity
	fileID, err := UserGetFileID(userdata, filename)
	if err != nil {
		return err
	}

	// Check user access to file
	if fileID == uuid.Nil {
		return errors.New("file access denied")
	}

	// Get File Struct
	fileStructBytes, ok := userlib.DatastoreGet(fileID)
	if !ok {
		return errors.New("file struct not found flag 1")
	}
	var oldFileStruct File
	err = json.Unmarshal(fileStructBytes, &oldFileStruct)
	if err != nil {
		return err
	}

	// // Check owner identity
	// if oldFileStruct.OwnerName != userdata.Username {
	// 	return errors.New("owner identity does not match")
	// }

	// Get file content key
	fileContentKey, err := UserGetFileContentKey(userdata, filename)
	if err != nil {
		return err
	}

	// Get Every Accessible username
	allUsername := []string{}

	// Get inv key struct
	encInvKeysBytes, ok := userlib.DatastoreGet(oldFileStruct.EncInvKeys)
	if !ok {
		return errors.New("inv key struct not found")
	}
	invKeysBytes, err := SymAuthDec(fileContentKey, encInvKeysBytes)
	if err != nil {
		return err
	}
	var invKeys InvKeys
	err = json.Unmarshal(invKeysBytes, &invKeys)
	if err != nil {
		return err
	}

	for _, usernameToKeyBytes := range invKeys.UsernameToInvKey {
		var usernameToKey UsernameInvKey
		err = json.Unmarshal(usernameToKeyBytes, &usernameToKey)
		if err != nil {
			return err
		}

		if !usernameIncluded(usernameToKey.SourceUsername, allUsername) {
			allUsername = append(allUsername, usernameToKey.SourceUsername)
		}

		if !usernameIncluded(usernameToKey.DestUsername, allUsername) {
			allUsername = append(allUsername, usernameToKey.DestUsername)
		}
	}

	// Check dest user access to file

	if !usernameIncluded(recipientUsername, allUsername) {
		return errors.New("recipient have no access to file")
	}

	// Get Every Revoking username
	revokeUsername := []string{recipientUsername}
	for _, usernameToKeyBytes := range invKeys.UsernameToInvKey {
		var usernameToKey UsernameInvKey
		err = json.Unmarshal(usernameToKeyBytes, &usernameToKey)
		if err != nil {
			return err
		}

		if usernameIncluded(usernameToKey.SourceUsername, revokeUsername) {
			revokeUsername = append(revokeUsername, usernameToKey.DestUsername)
		}
	}

	// Re-encrypt file with new content key
	fileContent, err := userdata.LoadFile(filename)
	if err != nil {
		return err
	}

	for _, filenameFileIDKey := range userdata.FileAccess {
		if filenameFileIDKey.Filename == filename {
			filenameFileIDKey.Filename = filename + " "

			//userlib.DatastoreDelete(oldFileStruct.ID)

			err = userdata.StoreFile(filename, fileContent)
			if err != nil {
				return err
			}

			filenameFileIDKey.Filename = filename
		}
	}

	// Get the new key
	newKey, err := UserGetFileContentKey(userdata, filename)
	if err != nil {
		return err
	}

	// Get new file struct
	fileStructBytes, ok = userlib.DatastoreGet(fileID)
	if !ok {
		return errors.New("file struct not found flag 2")
	}
	var newFileStruct File
	err = json.Unmarshal(fileStructBytes, &newFileStruct)
	if err != nil {
		return err
	}

	// Manually add the user to the file key manager
	for _, id := range oldFileStruct.FileKeyManagers {
		encFileKeyManager, ok := userlib.DatastoreGet(id)
		if !ok {
			return errors.New("key manager not found")
		}

		for _, usernameToKeyBytes := range invKeys.UsernameToInvKey {
			var usernameToKey UsernameInvKey
			err = json.Unmarshal(usernameToKeyBytes, &usernameToKey)
			if err != nil {
				return err
			}

			fileKeyManagerBytes, err := SymAuthDec(usernameToKey.InvitationKey, encFileKeyManager)
			if err == nil {
				var keyManager KeyManager
				err = json.Unmarshal(fileKeyManagerBytes, &keyManager)
				if err != nil {
					return err
				}

				if usernameToKey.DestUsername != userdata.Username {
					if usernameIncluded(usernameToKey.DestUsername, revokeUsername) {
						userlib.DatastoreDelete(usernameToKey.InvitationID)
						userlib.DatastoreDelete(keyManager.KeyManagerID)
					} else {
						keyManager.ContentKey = newKey

						fileKeyManagerBytes, err = json.Marshal(keyManager)
						if err != nil {
							return err
						}

						encFileKeyManager, err := SymAuthEnc(usernameToKey.InvitationKey, fileKeyManagerBytes)
						if err != nil {
							return err
						}

						userlib.DatastoreSet(keyManager.KeyManagerID, encFileKeyManager)

						newFileStruct.FileKeyManagers = append(newFileStruct.FileKeyManagers, keyManager.KeyManagerID)
					}
				}
			}
		}
	}

	// Update file struct with the new key manager
	fileBytes, err := json.Marshal(newFileStruct)
	if err != nil {
		return err
	}
	userlib.DatastoreSet(fileID, fileBytes)

	err = uploadUser(userdata)
	if err != nil {
		return err
	}

	return nil
}

// Helper Functions
func usernameIncluded(username string, namelist []string) bool {
	for _, name := range namelist {
		if username == name {
			return true
		}
	}
	return false
}

func UserGetFileAccess(userdata *User, filename string) (UUID, []byte, error) { // returns FKM_ID, inv key
	for _, filenameFileIDKey := range userdata.FileAccess {
		if filenameFileIDKey.Filename == filename {
			return filenameFileIDKey.KeyManagerID, filenameFileIDKey.InvitationKey, nil
		}
	}
	return uuid.Nil, nil, nil
}

func UserGetFileID(userdata *User, filename string) (UUID, error) { // returns FKM_ID, inv key
	for _, filenameFileIDKey := range userdata.FileAccess {
		if filenameFileIDKey.Filename == filename {
			return filenameFileIDKey.FileID, nil
		}
	}
	return uuid.Nil, nil
}

func UserGetFileContentKey(userdata *User, filename string) ([]byte, error) {
	// Get invitation key
	fileKeyManagerID, invitationKey, err := UserGetFileAccess(userdata, filename)
	if err != nil {
		return nil, err
	}

	// Get File Key Manager
	encKeyManagerDataBytes, ok := userlib.DatastoreGet(fileKeyManagerID)
	if !ok {
		return nil, errors.New("key manager not found from GetFileContentKey")
	}
	keyManagerDataBytes, err := SymAuthDec(invitationKey, encKeyManagerDataBytes)
	if err != nil {
		return nil, err
	}
	var keyManager KeyManager
	json.Unmarshal(keyManagerDataBytes, &keyManager)

	return keyManager.ContentKey, nil
}

// Login function
func Login(username, password string) ([]byte, bool) {
	// return DataStoreGet(uuid.FromBytes(H(username + password)))

	hashedCredentials := userlib.Argon2Key([]byte(username+password), []byte("salt"), 16)

	userUUID, err := uuid.FromBytes(hashedCredentials)
	if err != nil {
		return nil, false
	}
	return userlib.DatastoreGet(userUUID)
}

// Update user function
func downloadUser(user *User) error {
	// online_user = Login()
	// if user != online_user:
	//     user = online_user

	encOnlineUserData, found := Login(user.Username, user.Password)
	if !found {
		return errors.New("user not found")
	}

	onlineUserData, err := SymAuthDec(user.SymKey, encOnlineUserData)
	if err != nil {
		return err
	}

	var onlineUser User
	err = json.Unmarshal(onlineUserData, &onlineUser)
	if err != nil {
		return err
	}

	*user = onlineUser
	return nil

}

func uploadUser(user *User) error {
	userBytes, err := json.Marshal(user)
	if err != nil {
		return err
	}
	encUserBytes, err := SymAuthEnc(user.SymKey, userBytes)
	if err != nil {
		return err
	}
	userlib.DatastoreSet(user.ID, encUserBytes)

	return nil
}

func GetFile(user *User, filename string) (*FileContent, error) {

	// Get invitation key
	fileKeyManagerID, invitationKey, err := UserGetFileAccess(user, filename)
	if err != nil {
		return nil, err
	}
	if fileKeyManagerID == uuid.Nil {
		return nil, errors.New("key manager not exist in file access")
	}

	// Get File Key Manager
	encKeyManagerDataBytes, ok := userlib.DatastoreGet(fileKeyManagerID)
	if !ok {
		return nil, errors.New("key manager not found from Getfile")
	}
	keyManagerDataBytes, err := SymAuthDec(invitationKey, encKeyManagerDataBytes)
	if err != nil {
		return nil, err
	}
	var keyManager KeyManager
	json.Unmarshal(keyManagerDataBytes, &keyManager)

	fileContentKey, err := UserGetFileContentKey(user, filename)
	if err != nil {
		return nil, err
	}

	// Retrieve and decrypt the file content
	fileContentData, ok := userlib.DatastoreGet(keyManager.FileContentID)
	if !ok {
		return nil, errors.New("file content not found")
	}

	decryptedFileContent, err := SymAuthDec(fileContentKey, fileContentData)
	if err != nil {
		return nil, err
	}

	var fileContent FileContent
	err = json.Unmarshal(decryptedFileContent, &fileContent)
	if err != nil {
		return nil, err
	}

	return &fileContent, nil
}

func GetUserPubKey(username string) (userlib.PKEEncKey, userlib.DSVerifyKey, error) {

	publicEncKey, ok := userlib.KeystoreGet(username + "Enc")
	if !ok {
		return userlib.PKEEncKey{}, userlib.DSVerifyKey{}, errors.New("public key not found")
	}
	publicVerKey, ok := userlib.KeystoreGet(username + "Ver")
	if !ok {
		return userlib.PKEEncKey{}, userlib.DSVerifyKey{}, errors.New("public key not found")
	}

	return publicEncKey, publicVerKey, nil

}

func UserExist(username string) bool {
	_, ok := userlib.KeystoreGet(username + "Enc")
	return ok
}

// Symmetric authenticated encryption
func SymAuthEnc(k []byte, msg []byte) ([]byte, error) {

	// k1, k2 = HashKDF(k)

	// return CTR-Enc(key=k1, iv=rand(), msg)
	//     +HMAC(key=k2, msg)

	k1, err := userlib.HashKDF(k, []byte("encryption"))
	if err != nil {
		return nil, err
	}

	k2, err := userlib.HashKDF(k, []byte("integrity"))
	if err != nil {
		return nil, err
	}

	iv := userlib.RandomBytes(16)
	ciphertext := userlib.SymEnc(k1[:16], iv, msg)

	hmac, err := userlib.HMACEval(k2[:16], ciphertext)
	if err != nil {
		return nil, err
	}

	return append(ciphertext, hmac...), nil
}

func SymAuthDec(k []byte, ciphertext []byte) ([]byte, error) {
	if len(ciphertext) < 80 {
		return nil, errors.New("ciphertext too short")
	}
	hmac := ciphertext[len(ciphertext)-64:]
	ciphertext = ciphertext[:len(ciphertext)-64]

	k1, err := userlib.HashKDF(k, []byte("encryption"))
	if err != nil {
		return nil, err
	}

	k2, err := userlib.HashKDF(k, []byte("integrity"))
	if err != nil {
		return nil, err
	}

	msg := userlib.SymDec(k1[:16], ciphertext)

	// HMAC Verification
	generated_hmac, err := userlib.HMACEval(k2[:16], ciphertext)
	if err != nil {
		return nil, err
	}
	if !userlib.HMACEqual(generated_hmac, hmac) {
		return nil, errors.New("hmac verification failed")
	}

	return msg, nil
}

func AssymAuthEnc(publicEncKey userlib.PKEEncKey, privateSignKey userlib.DSSignKey, msg []byte) ([]byte, error) {
	ciphertext, err := userlib.PKEEnc(publicEncKey, msg)
	if err != nil {
		return nil, err
	}

	signature, err := userlib.DSSign(privateSignKey, msg)
	if err != nil {
		return nil, err
	}

	return append(ciphertext, signature...), nil
}

func AssymAuthDec(privateDecKey userlib.PKEDecKey, publicVerKey userlib.DSVerifyKey, ciphertext []byte) ([]byte, error) {
	if len(ciphertext) < 256 {
		return nil, errors.New("ciphertext too short")
	}

	signature := ciphertext[len(ciphertext)-256:]
	ciphertext = ciphertext[:len(ciphertext)-256]

	msg, err := userlib.PKEDec(privateDecKey, ciphertext)
	if err != nil {
		return nil, err
	}

	// RSA Verification
	err = userlib.DSVerify(publicVerKey, msg, signature)
	if err != nil {
		return nil, err
	}

	return msg, nil
}

func HybridEnc(publicEncKey userlib.PKEEncKey, privateSignKey userlib.DSSignKey, msg []byte) ([]byte, error) {

	// Generate a randomsymmetric key
	symKey := userlib.RandomBytes(16)

	// Symmetrically encrypt the message using the symmetric key
	encMsg, err := SymAuthEnc(symKey, msg)
	if err != nil {
		return nil, err
	}

	// Asymmetrically encrypt the symmetric keyusing the recipient's public encryption key
	encSymKey, err := AssymAuthEnc(publicEncKey, privateSignKey, symKey)
	if err != nil {
		return nil, err
	}

	return append(encSymKey, encMsg...), nil
}

// assymmetric encrypted key (16 bytes key and 256 bytes signature)
// symmetric encrypted ciphertext msg

func HybridDec(privateDecKey userlib.PKEDecKey, publicVerKey userlib.DSVerifyKey, ciphertext []byte) (msg []byte, err error) {
	if len(ciphertext) < 512 {
		return nil, errors.New("too short cipher text")
	}

	// Get Symmetric Key
	symKey, err := AssymAuthDec(privateDecKey, publicVerKey, ciphertext[:512])
	if err != nil {
		return nil, err
	}

	// Decrypt content
	msg, err = SymAuthDec(symKey, ciphertext[512:])
	if err != nil {
		return nil, err
	}

	return msg, nil
}

func extendString(str string, length int) string {
	if len(str) >= length {
		return str[:length]
	}
	return str + strings.Repeat(" ", length-len(str))
}

func handleConnection(conn net.Conn) {
	defer conn.Close()

	reader := bufio.NewReader(conn)
	for {
		// Read message from the client
		message, err := reader.ReadString(0x00) // 0x00 is the null terminator
		if err != nil {
			if err == io.EOF {
				fmt.Println("Client disconnected")
				return
			}
			fmt.Println("Error reading from client:", err)
			return
		}

		fmt.Println("Received: ", message)

		response := ""

		opcode := message[0]
		fmt.Printf("Opcode: %d\n", opcode)
		switch opcode {
		case '\x01':
			// InitUser
			fmt.Println("InitUser")
			username := strings.ReplaceAll(message[1:33], string(0x01), "")
			password := strings.ReplaceAll(message[33:65], string(0x01), "")
			user, err := InitUser(username, password)
			if err == nil {
				response = "User initialized: " + user.Username
			} else {
				response = "User initialization failed: " + err.Error()
			}
		case '\x02':
			// GetUser
			fmt.Println("GetUser")
			username := strings.ReplaceAll(message[1:33], string(0x01), "")
			password := strings.ReplaceAll(message[33:65], string(0x01), "")
			user, err := GetUser(username, password)
			if err == nil {
				response = "User retrieved: " + user.Username
			} else {
				response = "User retrieving failed: " + err.Error()
			}
		case '\x03':
			// StoreFile
			fmt.Println("StoreFile")
			username := strings.ReplaceAll(message[1:33], string(0x01), "")
			password := strings.ReplaceAll(message[33:65], string(0x01), "")
			user, err := GetUser(username, password)
			if err != nil {
				response = "File storing failed: " + err.Error()
			} else {
				filename := strings.ReplaceAll(message[65:97], string(0x01), "")
				filedata := strings.ReplaceAll(message[97:], string(0x01), "")
				filedata = filedata[:len(filedata)-1] // Remove the null terminator
				err = user.StoreFile(filename, []byte(filedata))
				if err == nil {
					response = "File stored: " + filename
				} else {
					response = "File storing failed: " + err.Error()
				}
			}
		case '\x04':
			// AppendToFile
			fmt.Println("AppendToFile")
			username := strings.ReplaceAll(message[1:33], string(0x01), "")
			password := strings.ReplaceAll(message[33:65], string(0x01), "")
			user, err := GetUser(username, password)
			if err != nil {
				response = "File appending failed: " + err.Error()
			} else {
				filename := strings.ReplaceAll(message[65:97], string(0x01), "")
				filedata := strings.ReplaceAll(message[97:], string(0x01), "")
				filedata = filedata[:len(filedata)-1] // Remove the null terminator
				err = user.AppendToFile(filename, []byte(filedata))
				if err == nil {
					response = "File appended: " + filename
				} else {
					response = "File appending failed: " + err.Error()
				}
			}
		case '\x05':
			// LoadFile
			fmt.Println("LoadFile")
			username := strings.ReplaceAll(message[1:33], string(0x01), "")
			password := strings.ReplaceAll(message[33:65], string(0x01), "")
			user, err := GetUser(username, password)
			if err != nil {
				response = "File loading failed: " + err.Error()
			} else {
				filename := strings.ReplaceAll(message[65:97], string(0x01), "")
				content, err := user.LoadFile(filename)
				if err == nil {
					response = strings.ReplaceAll(string(content), string(0x00), "")
				} else {
					response = "File loading failed: " + err.Error()
				}
			}
		case '\x06':
			// CreateInvitation
			fmt.Println("CreateInvitation")
			username := strings.ReplaceAll(message[1:33], string(0x01), "")
			password := strings.ReplaceAll(message[33:65], string(0x01), "")
			user, err := GetUser(username, password)
			if err != nil {
				response = "Invitation creation failed: " + err.Error()
			} else {
				filename := strings.ReplaceAll(message[65:97], string(0x01), "")
				receipientUsername := strings.ReplaceAll(message[97:129], string(0x01), "")
				invitation, err := user.CreateInvitation(filename, receipientUsername)
				if err == nil {
					response = invitation.String()
				} else {
					response = "Invitation creation failed: " + err.Error()
				}
			}
		case '\x07':
			// AcceptInvitation
			fmt.Println("AcceptInvitation")
			username := strings.ReplaceAll(message[1:33], string(0x01), "")
			password := strings.ReplaceAll(message[33:65], string(0x01), "")
			user, err := GetUser(username, password)
			if err != nil {
				response = "Invitation acceptance failed: " + err.Error()
			} else {
				senderName := strings.ReplaceAll(message[65:97], string(0x01), "")
				filename := strings.ReplaceAll(message[97:129], string(0x01), "")
				invitationID := strings.ReplaceAll(message[129:], string(0x01), "")
				invitationID = invitationID[:len(invitationID)-1] // Remove the null terminator
				invitationUUID, err := uuid.Parse(invitationID)
				if err == nil {
					err = user.AcceptInvitation(senderName, invitationUUID, filename)
					if err != nil {
						response = "Invitation acceptance failed: " + err.Error()
					} else {
						response = "Invitation accepted: " + filename
					}
				} else {
					response = "Invitation acceptance failed: " + err.Error()
				}
			}
		case '\x08':
			// RevokeAccess
			fmt.Println("RevokeAccess")
			username := strings.ReplaceAll(message[1:33], string(0x01), "")
			password := strings.ReplaceAll(message[33:65], string(0x01), "")
			user, err := GetUser(username, password)
			if err != nil {
				response = "Access revoking failed: " + err.Error()
			} else {
				filename := strings.ReplaceAll(message[65:97], string(0x01), "")
				receipientUsername := strings.ReplaceAll(message[97:129], string(0x01), "")
				err = user.RevokeAccess(filename, receipientUsername)
				if err == nil {
					response = "Access revoked: " + filename
				} else {
					response = "Access revoking failed: " + err.Error()
				}
			}
		default:
		}

		if response != "" {
			fmt.Println("Response: ", response)
			response += string(0x00)
			_, err = conn.Write([]byte(response))
			if err != nil {
				fmt.Println("Error writing to client:", err)
				return
			}
		}
	}
}

func main() {
	// Start listening on a specific port
	listener, err := net.Listen("tcp", ":8080")
	if err != nil {
		fmt.Println("Error starting server:", err)
		return
	}
	defer listener.Close()
	fmt.Println("Server is listening on port 8080...")

	for {
		// Accept a new connection
		conn, err := listener.Accept()
		if err != nil {
			fmt.Println("Error accepting connection:", err)
			continue
		}
		fmt.Println("New client connected!")

		// Handle the connection in a separate goroutine
		go handleConnection(conn)
	}
}
