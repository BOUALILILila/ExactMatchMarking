
import logging
import tensorflow as tf
import os

logger = logging.getLogger(__name__)

class TFRecordHandle(object):

    def __init__(
        self, 
        tokenizer, 
        max_seq_length, 
        max_query_length, 
        seed=42,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.seed = seed

        if max_seq_length> 512 :
            raise ValueError('Max sequence length must be < 512 to avoid indexing errors.')
        if max_query_length < 2 :
            raise ValueError('Max query length must be at least 2 to include [CLS] and [SEP].')
        if max_query_length > max_seq_length:
            raise ValueError('Max sequence length must be > to max query length.')

        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length

    def get_train_dataset(self, data_path, batch_size):
        raise NotImplementedError()
    
    def get_eval_dataset(self, data_path, batch_size, num_skip=0):
        dataset = tf.data.TFRecordDataset([data_path])
        dataset = dataset.map( lambda record : self._extract_fn_eval(record)).prefetch(batch_size*1000)
        if num_skip > 0:
            dataset = dataset.skip(num_skip)
        count = dataset.reduce(0, lambda x, _: x + 1)
        dataset = dataset.padded_batch(
                    batch_size=batch_size,
                    padded_shapes=({
                        "id" : [],
                        "input_ids": [self.max_seq_length],
                        "attention_mask": [self.max_seq_length],
                        "token_type_ids": [self.max_seq_length],
                        "len_gt_titles": []
                        }, []
                    ),
                    padding_values=({
                        "id": 0,
                        "input_ids": 0,
                        "attention_mask": 0,
                        "token_type_ids": 0,
                        "len_gt_titles": 0
                        }, 0
                    ),
                    drop_remainder=False)
        return dataset, count.numpy()
         
    
    def write_train_example(
                self, 
                tf_writer,
                query,
                docs,
                labels
    ):
        raise NotImplementedError()
    
    def write_eval_example(
                self, 
                tf_writer,
                query,
                docs,
                labels,
                query_id, 
                doc_ids,
                len_gt
    ):
        raise NotImplementedError()
    
    def _extract_fn_train(self,data_record):
        raise NotImplementedError()
    
    def _extract_fn_eval(self,data_record):
        raise NotImplementedError()


class PassageHandle(TFRecordHandle):
    def __init__(
        self, 
        tokenizer, 
        max_seq_length=512, 
        max_query_length=64, 
        seed=42,
        **kwargs
    ):
        super(PassageHandle,self).__init__(tokenizer, max_seq_length, max_query_length, seed, **kwargs)
        
    def get_train_dataset (self, data_path, batch_size):
        dataset = tf.data.TFRecordDataset([data_path])
        dataset = dataset.map( lambda record : self._extract_fn_train(record)).prefetch(batch_size*1000)
        count = dataset.reduce(0, lambda x, _: x + 1)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=1000, seed=self.seed)
        dataset = dataset.padded_batch(
                    batch_size=batch_size,
                    padded_shapes=({
                        "input_ids": [self.max_seq_length],
                        "attention_mask": [self.max_seq_length],
                        "token_type_ids": [self.max_seq_length]},
                        []
                    ),
                    padding_values=({
                        "input_ids": 0,
                        "attention_mask": 0,
                        "token_type_ids": 0},
                        0
                    ),
                    drop_remainder=True)
        return dataset, count.numpy() 

    def _extract_fn_train(self,data_record):
        features = {
          "query_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "doc_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "label": tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)
        
        query_ids = tf.cast(sample["query_ids"], tf.int32) # max length with special tokens
        doc_ids = tf.cast(sample["doc_ids"], tf.int32) #max length with special tokens
        label_ids = tf.cast(sample["label"], tf.int32)
        
        input_ids, input_mask, segment_ids = self._encode(query_ids, doc_ids)

        features = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
        }
        
        return (features, label_ids)

    def _extract_fn_eval(self, data_record):
        features = {
          "id" : tf.io.FixedLenFeature([], tf.int64),
          "query_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "doc_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "label": tf.io.FixedLenFeature([], tf.int64),
          "len_gt_titles": tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)

        id_pair = tf.cast(sample["id"], tf.int32)
        query_ids = tf.cast(sample["query_ids"], tf.int32)  
        doc_ids = tf.cast(sample["doc_ids"], tf.int32) 
        label_ids = tf.cast(sample["label"], tf.int32)
        len_gt_titles = tf.cast(sample["len_gt_titles"], tf.int32)
        
        input_ids, input_mask, segment_ids = self._encode(query_ids, doc_ids)

        features = {
            "id" : id_pair,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
            "len_gt_titles": len_gt_titles,
        }
        
        return (features, label_ids)

    
    def _encode(self, query_ids, doc_ids):
        query_ids_without_sep = query_ids[:-1]
        query_ids_trunc = tf.concat((query_ids_without_sep[:self.max_query_length-1], query_ids[-1:]),0) # add SEP end
        doc_ids_without_markers = doc_ids[1:-1] # remove CLS and SEP
        
        input_ids = tf.concat((query_ids_trunc, doc_ids_without_markers), 0) # need [SEP] at last and cut at 512
        input_ids = tf.concat((input_ids[:self.max_seq_length-1],doc_ids[-1:]), 0)

        input_mask = tf.ones_like(input_ids)

        query_segment_id = tf.zeros_like(query_ids_trunc)
        doc_segment_id = tf.ones_like(doc_ids[1:])
        segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)
        segment_ids = segment_ids[:self.max_seq_length]
        return input_ids, input_mask, segment_ids

    def write_train_example(self, tf_writer,
                                         query,
                                         docs,
                                         labels):
        query_ids = self.tokenizer.encode(
                query,
                add_special_tokens = True,
                max_length = self.max_seq_length,
                truncation = True,
            )

        query_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=query_ids))
                        
        for i, (doc_text, label) in enumerate(zip(docs, labels)):
            doc_ids = self.tokenizer.encode(
                    doc_text,
                    add_special_tokens = True,
                    max_length = self.max_seq_length,
                    truncation= True,
                )

            doc_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=doc_ids))

            labels_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label]))

            features = tf.train.Features(feature={
                'query_ids': query_ids_tf,
                'doc_ids': doc_ids_tf,
                'label': labels_tf,
            })
            example = tf.train.Example(features=features)
            tf_writer.write(example.SerializeToString())
    
    def write_eval_example(self, tf_writer, ids_writer, i_ids,
                                            query,
                                            docs,
                                            labels,
                                            query_id, 
                                            doc_ids,
                                            len_gt):

        query_ids = self.tokenizer.encode(
                    query,
                    add_special_tokens=True,
                    max_length= self.max_query_length,
                    truncation= True,
                )

        query_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=query_ids))
        
        len_gt_titles_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[len_gt]))
                        
        for i, (doc_id, doc_text, label) in enumerate(zip(doc_ids, docs, labels)):
            
            doc_ids = self.tokenizer.encode(
                    doc_text,
                    add_special_tokens = True,
                    max_length = self.max_seq_length,
                    truncation= True,
                ) 

            doc_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=doc_ids))

            labels_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label]))
            
            id_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[i_ids]))
            
            features = tf.train.Features(feature={
                        'id' : id_tf,
                        'query_ids' : query_ids_tf,
                        'doc_ids': doc_ids_tf,
                        'label': labels_tf,
                        'len_gt_titles': len_gt_titles_tf,
                    })

            example = tf.train.Example(features=features)
            tf_writer.write(example.SerializeToString())
            ids_writer.write("\t".join([str(i_ids), query_id, doc_id, '0'])+"\n")
            i_ids += 1
        return i_ids

class DocumentHandle(TFRecordHandle):

    def __init__(
        self, 
        tokenizer, 
        max_seq_length=512, 
        max_query_length=64,  
        seed=42, 
        **kwargs
    ):
        super(DocumentHandle,self).__init__(tokenizer, max_seq_length, max_query_length, seed, **kwargs)

        self.max_title_length = kwargs['max_title_length'] if 'max_title_length' in kwargs.keys() else 64
        self.chunk_size = kwargs['chunk_size'] if 'chunk_size' in kwargs.keys() else 384
        self.stride = kwargs['stride'] if 'stride' in kwargs.keys() else 384
        if self.stride < 1 :
            raise ValueError('Stride musbt at least 1 token.')
        elif self.stride > self.chunk_size:
            raise ValueError('Stride can\'t be > to chunk size.')

        
    
    
    def write_eval_example(self, tf_writer, ids_writer, i_ids,
                                            query,
                                            docs,
                                            labels,
                                            query_id, 
                                            doc_ids,
                                            len_gt):
        # q_id_tf = tf.train.Feature(
        #             bytes_list=tf.train.BytesList(value=[query_id.encode()]))

        query_ids = self.tokenizer.encode(
                    query,
                    add_special_tokens=True,
                    max_length= self.max_query_length,
                    truncation = True,
                )
        query_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=query_ids))
        
        len_gt_titles_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[len_gt]))
                        
        for i, (doc_id, doc_tuple, label) in enumerate(zip(doc_ids, docs, labels)):
            doc_title, doc_text = doc_tuple

            # d_id_tf = tf.train.Feature(
            #             bytes_list=tf.train.BytesList(value=[doc_id.encode()]))

            title_ids = self.tokenizer.encode(
                    doc_title,
                    add_special_tokens=False,
                    max_length= self.max_title_length,
                    truncation= True,
                )

            title_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=title_ids))
            
            doc_ids = self.tokenizer.encode(
                    doc_text,
                    add_special_tokens=False,
                    truncation = False,
                )  
            
            labels_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label]))
            
            if len(doc_ids)+len(title_ids)+len(query_ids)+1 > self.max_seq_length:
                i = 0
                while len(doc_ids)>0:
                    passage_ids = doc_ids[:self.chunk_size-1]
                    pass_id = f'{doc_id}_{i}'
                    i += 1
                    
                    doc_ids_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=passage_ids))
                    
                    id_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[i_ids]))

                    features = tf.train.Features(feature={
                        'id' : id_tf,
                        'query_ids' : query_ids_tf,
                        'title_ids' : title_ids_tf,
                        'doc_ids': doc_ids_tf,
                        'label': labels_tf,
                        'len_gt_titles': len_gt_titles_tf,
                    })

                    example = tf.train.Example(features=features)
                    tf_writer.write(example.SerializeToString())
                    ids_writer.write("\t".join([str(i_ids),query_id, pass_id])+"\n")
                    i_ids += 1
                    doc_ids = doc_ids[self.chunk_size:]

            else: 
                doc_ids_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=doc_ids))
                id_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[i_ids]))
                features = tf.train.Features(feature={
                        'id' : id_tf,
                        'query_ids' : query_ids_tf,
                        'title_ids' : title_ids_tf,
                        'doc_ids': doc_ids_tf,
                        'label': labels_tf,
                        'len_gt_titles': len_gt_titles_tf,
                    })
                example = tf.train.Example(features=features)
                tf_writer.write(example.SerializeToString())
                ids_writer.write("\t".join([str(i_ids), query_id, doc_id])+"\n")
                i_ids += 1
        return i_ids
    
    def _encode(self, query_ids, title_ids, doc_ids):
        d_len = self.max_seq_length - self.max_query_length - self.max_title_length -1
        document_ids = tf.concat(( title_ids, doc_ids[:d_len], query_ids[-1:]), axis= 0) # query_ids[-1:] == [SEP]
        input_ids = tf.concat((query_ids, document_ids), axis= 0)

        input_mask = tf.ones_like(input_ids)

        query_segment_id = tf.zeros_like(query_ids)
        doc_segment_id = tf.ones_like(document_ids)
        segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)

        return input_ids, input_mask, segment_ids
    
    def _extract_fn_eval(self, data_record):
        features = {
          "id" : tf.io.FixedLenFeature([], tf.int64),
          "query_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "title_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "doc_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "label": tf.io.FixedLenFeature([], tf.int64),
          "len_gt_titles": tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)

        id_pair = tf.cast(sample["id"], tf.int32)
        query_ids = tf.cast(sample["query_ids"], tf.int32) 
        title_ids = tf.cast(sample["title_ids"], tf.int32) 
        doc_ids = tf.cast(sample["doc_ids"], tf.int32) 
        label_ids = tf.cast(sample["label"], tf.int32)
        len_gt_titles = tf.cast(sample["len_gt_titles"], tf.int32)
        
        input_ids, input_mask, segment_ids = self._encode(query_ids, title_ids, doc_ids)

        features = {
            "id" : id_pair,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
            "len_gt_titles": len_gt_titles,
        }
        
        return (features, label_ids)
    
class DocumentSplitterHandle(DocumentHandle):

    def __init__(
        self, 
        tokenizer,
        max_seq_length=512, 
        max_query_length=64,  
        seed=42, 
        **kwargs
    ):
        super(DocumentSplitterHandle,self).__init__(tokenizer, max_seq_length, max_query_length, seed, **kwargs)
        self.stride = kwargs['stride'] if 'stride' in kwargs.keys() else 192

    def write_eval_example(self, tf_writer, ids_writer, i_ids,
                                query,
                                docs,
                                labels,
                                query_id, 
                                doc_ids,
                                len_gt):

        query_ids = self.tokenizer.encode(
                    query,
                    add_special_tokens = True,
                    max_length = self.max_query_length,
                    truncation= True,
                )

        query_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=query_ids))
        
        len_gt_titles_tf = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[len_gt]))
                        
        for i, (doc_id, doc_tuple, label) in enumerate(zip(doc_ids, docs, labels)):
            doc_title, doc_text = doc_tuple

            title_ids = self.tokenizer.encode(
                    doc_title,
                    add_special_tokens=False,
                    max_length=self.max_title_length,
                    truncation= True,
                )

            title_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=title_ids))

            labels_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label]))
            
            doc_ids = self.tokenizer.encode(
                    doc_text,
                    add_special_tokens=False,
                    truncation = False,
                )
             
            i = 0
            while len(doc_ids)>0:
                passage_ids = doc_ids[:self.chunk_size-1] # 1 token for the [SEP]
                #passage_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(passage_ids))
                
                doc_ids_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=passage_ids))
                
                id_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[i_ids]))
                
                features = tf.train.Features(feature={
                    'id' : id_tf,
                    'query_ids' : query_ids_tf,
                    'title_ids' : title_ids_tf,
                    'doc_ids': doc_ids_tf,
                    'label': labels_tf,
                    'len_gt_titles': len_gt_titles_tf,
                })

                example = tf.train.Example(features=features)
                tf_writer.write(example.SerializeToString())
                ids_writer.write("\t".join([str(i_ids), query_id, doc_id, str(i)])+"\n")
                i_ids += 1
                i += 1

                doc_ids = doc_ids[self.stride:]
                if len(doc_ids)<self.stride:
                    break
                # end of the doc_ids no more data
            #next document
        return i_ids

    def write_train_example(self, tf_writer,
                                query,
                                docs,
                                labels):

        query_ids = self.tokenizer.encode(
                    query,
                    add_special_tokens=True,
                    max_length= self.max_query_length,
                    truncation = True,
                )
        
        query_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=query_ids))
                                
        for i, (doc_tuple, label) in enumerate(zip(docs, labels)):
            doc_title, doc_text = doc_tuple

            title_ids = self.tokenizer.encode(
                    doc_title,
                    add_special_tokens=False,
                    max_length=self.max_title_length, 
                    truncation = True,
                )
              
            title_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=title_ids))

            labels_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label]))
            
            doc_ids = self.tokenizer.encode(
                    doc_text,
                    add_special_tokens=False,
                    truncation  = False,
                ) 

            while len(doc_ids)>0:
                passage_ids = doc_ids[:self.chunk_size-1]
                
                doc_ids_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=passage_ids))
                
                features = tf.train.Features(feature={
                    'query_ids' : query_ids_tf,
                    'title_ids' : title_ids_tf,
                    'doc_ids': doc_ids_tf,
                    'label': labels_tf,
                })

                example = tf.train.Example(features=features)
                tf_writer.write(example.SerializeToString())
                

                doc_ids = doc_ids[self.stride:]
                if len(doc_ids)<self.stride:
                    break
                # end of the doc_ids no more data
            #next document


    def _extract_fn_train(self, data_record):
        features = {
          "query_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "title_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "doc_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "label": tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)

        query_ids = tf.cast(sample["query_ids"], tf.int32) 
        title_ids = tf.cast(sample["title_ids"], tf.int32) 
        doc_ids = tf.cast(sample["doc_ids"], tf.int32) 
        label_ids = tf.cast(sample["label"], tf.int32)
        
        input_ids, input_mask, segment_ids = self._encode(query_ids, title_ids, doc_ids)

        features = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
        }
        
        return (features, label_ids)

    def get_train_dataset (self, data_path, batch_size):
        dataset = tf.data.TFRecordDataset([data_path])
        dataset = dataset.map( lambda record : self._extract_fn_train(record)).prefetch(batch_size*1000)
        count = dataset.reduce(0, lambda x, _: x + 1)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=1000, seed=self.seed)
        dataset = dataset.padded_batch(
                    batch_size=batch_size,
                    padded_shapes=({
                        "input_ids": [self.max_seq_length],
                        "attention_mask": [self.max_seq_length],
                        "token_type_ids": [self.max_seq_length]},
                        []
                    ),
                    padding_values=({
                        "input_ids": 0,
                        "attention_mask": 0,
                        "token_type_ids": 0},
                        0
                    ),
                    drop_remainder=True)
        return dataset, count.numpy() #count


